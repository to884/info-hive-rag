"""wiki の PDF 取り込みを起点とするローカル索引。回答生成は行わない。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from datetime import timedelta
from contextlib import closing, contextmanager
from pathlib import Path

from filelock import FileLock

from rag_config import Config, write_json
from rag_pdf import chunks_for, extract_pages


def _same_file(first: os.stat_result, second: os.stat_result) -> bool:
    return (first.st_dev, first.st_ino, first.st_size, first.st_mtime_ns) == (
        second.st_dev, second.st_ino, second.st_size, second.st_mtime_ns)


def read_pdf(path: Path) -> tuple[bytes, os.stat_result]:
    """確かめたファイルを 1 回だけ開いて読み、読んだバイト列が PDF であることを確かめる（ADR-0013）。"""
    before = os.lstat(path)
    with open(path, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if not _same_file(before, opened):
            raise ValueError("確かめた PDF と開いたファイルが違います。もう一度取り込んでください。")
        data = handle.read()
        after = os.fstat(handle.fileno())
    if not _same_file(opened, after) or len(data) != after.st_size:
        raise ValueError("読み込み中に PDF が変更されました。もう一度取り込んでください。")
    if not data.startswith(b"%PDF-"):
        raise ValueError("PDF の形式ではありません（先頭が %PDF- ではない）。")
    return data, after


class Embedder:
    def __init__(self, config: Config):
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(config.threads)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        self.model = SentenceTransformer(config.model, revision=config.revision, device="cpu",
                                         cache_folder=str(config.model_cache_dir),
                                         model_kwargs={"attn_implementation": "sdpa"})
        # モデル本来の文脈長を使い、トークン数は encode 前に検査する。
        native_limit=int(self.model[0].auto_model.config.max_position_embeddings)
        self.model.max_seq_length=min(native_limit,8192)
        self.dimension = config.dimension

    def encode(self, texts: list[str], *, query: bool = False):
        prefix = "検索クエリ: " if query else "検索文書: "
        inputs=[prefix+text for text in texts]
        if not inputs:
            return self.model.encode(inputs,normalize_embeddings=True,show_progress_bar=False,convert_to_numpy=True)
        lengths=[len(ids) for ids in self.model.tokenizer(inputs,truncation=False,padding=False)['input_ids']]
        longest=max(lengths)
        if longest>self.model.max_seq_length:
            raise ValueError(f"埋め込み入力がモデルの上限 {self.model.max_seq_length} トークンを超えています。本文を失わないよう分割を小さくしてください。")
        # 長い入力で注意行列のメモリが膨らまないよう、二乗長に応じて件数を下げる。
        batch=max(1,min(32,len(inputs),(8*2048**2)//max(1,longest**2)))
        return self.model.encode(inputs, normalize_embeddings=True,batch_size=batch,
                                 show_progress_bar=False, convert_to_numpy=True)


class Engine:
    def __init__(self, config: Config | None = None, embedder=None, extractor=None, reranker=None):
        self.config = config or Config()
        self._embedder = embedder
        self.extractor = extractor or extract_pages
        self._tokenizer = None
        self._reranker = reranker

    @property
    def reranker(self):
        if self._reranker is None:
            from rag_rerank import Reranker
            self._reranker = Reranker(self.config)
        return self._reranker

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = Embedder(self.config)
        return self._embedder

    def tokenize(self, text: str) -> str:
        import unicodedata
        from sudachipy import dictionary, tokenizer

        if self._tokenizer is None:
            self._tokenizer = dictionary.Dictionary().create()
        normalized = unicodedata.normalize("NFKC", text).lower()
        return " ".join(word.normalized_form().lower() for word in
                        self._tokenizer.tokenize(normalized, tokenizer.Tokenizer.SplitMode.C)
                        if word.surface().strip() and word.part_of_speech()[0] != "補助記号")

    @contextmanager
    def _catalog(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(folder / "catalog.sqlite")
        connection.row_factory = sqlite3.Row
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY, generation TEXT NOT NULL, title TEXT NOT NULL,
                pages INTEGER, chunks INTEGER, fingerprint TEXT NOT NULL, ingested_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS locations (
                path TEXT PRIMARY KEY, doc_id TEXT NOT NULL, size INTEGER, mtime_ns INTEGER,
                confirmed_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS citations (
                doc_id TEXT, source_page TEXT, PRIMARY KEY(doc_id, source_page));
            CREATE TABLE IF NOT EXISTS jobs (
                generation TEXT PRIMARY KEY, completed INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'pending', error TEXT);
        """)
        with closing(connection), connection:
            yield connection

    def _table(self, folder: Path, create: bool = False):
        import lancedb
        import pyarrow as pa

        database = lancedb.connect(str(folder / "lance"), read_consistency_interval=timedelta(0))
        try:
            return database.open_table("chunks")
        except (FileNotFoundError, ValueError):
            if not create:
                return None
        schema = pa.schema([
            ("chunk_id", pa.string()), ("doc_id", pa.string()), ("generation", pa.string()),
            ("page", pa.int32()), ("section", pa.string()), ("title", pa.string()),
            ("text", pa.string()), ("terms", pa.string()),
            ("vector", pa.list_(pa.float32(), self.config.dimension)),
        ])
        return database.create_table("chunks", schema=schema, exist_ok=True)

    def _check_policy(self, folder: Path) -> None:
        path = folder / "policy.json"
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous["fingerprint"] != self.config.fingerprint:
                raise ValueError("索引と設定が異なります。モデルや分割方法の変更には全 PDF の再投入が必要です。別の data_dir で構築してください。")
        else:
            write_json(path, {"fingerprint": self.config.fingerprint, "model": self.config.model,
                              "revision": self.config.revision, "dimension": self.config.dimension})

    @staticmethod
    def _location(connection, doc_id: str) -> str | None:
        """文書の今の場所を返す。最後に確かめた順に見て、記録した大きさと更新時刻が今と合う場所を選ぶ。"""
        rows = connection.execute("SELECT path,size,mtime_ns FROM locations WHERE doc_id=? ORDER BY confirmed_at DESC",
                                  (doc_id,)).fetchall()
        for row in rows:
            try:
                current = os.stat(row["path"])
            except OSError:
                continue
            if (current.st_size, current.st_mtime_ns) == (row["size"], row["mtime_ns"]):
                return row["path"]
        return rows[0]["path"] if rows else None

    def _source_page(self, wiki: Path, source_page: str) -> str:
        return self.config.source_page(wiki, source_page)

    def _maintain_indexes(self,folder,table,progress=None):
        from rag_indexes import Indexes
        return Indexes(self.config,folder,table,progress).maintain()

    def maintain_indexes(self,wiki_root: str,progress=None):
        """既存の行を保持したまま索引を補修する。PDF 取り込みも同じ処理を呼ぶ。"""
        folder=self.config.workspace(wiki_root)
        with FileLock(str(folder/'write.lock'),timeout=1800):
            self._check_policy(folder)
            table=self._table(folder)
            return self._maintain_indexes(folder,table,progress) if table is not None else {'ann_ready':False,'rows':0}

    def _mark_reviews(self, wiki: Path, connection, doc_id: str) -> list[str]:
        import yaml

        direct = {row[0] for row in connection.execute("SELECT source_page FROM citations WHERE doc_id=?", (doc_id,))}
        slugs = {Path(page).stem for page in direct}
        affected = []
        for page in wiki.rglob("*.md"):
            if not page.resolve().is_relative_to(wiki) or any(part.startswith(".") for part in page.relative_to(wiki).parts):
                continue
            text = page.read_text(encoding="utf-8-sig")
            match = re.match(r"\A---\r?\n(.*?)\r?\n---(?:\r?\n|$)", text, re.S)
            if not match:
                continue
            meta = yaml.safe_load(match.group(1)) or {}
            if not isinstance(meta, dict):
                continue
            sources = meta.get("sources", meta.get("出典", []))
            if isinstance(sources, str):
                sources = [sources]
            sources = {str(value).strip("[]").split("|")[0] for value in sources or []}
            relative = page.relative_to(wiki).as_posix()
            if relative not in direct and not (sources & slugs):
                continue
            frontmatter = match.group(1)
            for key, value in {"review_required": "true", "review_reason": json.dumps("参照する PDF の内容が変わりました。出典を確認してください。", ensure_ascii=False)}.items():
                pattern = rf"(?m)^{key}:.*$"
                if re.search(pattern, frontmatter):
                    frontmatter = re.sub(pattern, f"{key}: {value}", frontmatter)
                else:
                    frontmatter += f"\n{key}: {value}"
            replacement = "---\n" + frontmatter + "\n---\n" + text[match.end():]
            temporary = page.with_suffix(".md.rag-tmp")
            temporary.write_text(replacement, encoding="utf-8")
            temporary.replace(page)
            affected.append(relative)
        return affected

    def ingest_pdf(self, wiki_root: str, source: str, source_page: str, title: str = "", progress=None,
                   before_publish=None) -> dict:
        wiki = Path(wiki_root).expanduser().resolve()
        folder = self.config.workspace(str(wiki))
        path = self.config.pdf_path(source)
        location = str(path)
        source_page = self._source_page(wiki, source_page)
        folder.mkdir(parents=True, exist_ok=True)
        with FileLock(str(folder / "write.lock"), timeout=1800):
            self._check_policy(folder)
            with self._catalog(folder) as connection:
                if progress:
                    progress("PDF の読み込み", 0, 1)
                # 遅い原資料を一度だけ順次読み込み、ハッシュと PDFium で共用する。
                data, stat_after = read_pdf(path)
                if progress:
                    progress("PDF の読み込み", 1, 1)
                doc_id = hashlib.sha256(data).hexdigest()
                generation = hashlib.sha256((doc_id + self.config.fingerprint).encode("ascii")).hexdigest()
                previous_location = connection.execute("SELECT doc_id FROM locations WHERE path=?", (location,)).fetchone()
                existing = connection.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
                old_id = previous_location[0] if previous_location and previous_location[0] != doc_id else None
                if existing:
                    pages_count, chunks_count = existing["pages"], existing["chunks"]
                    markdown = folder / "extracted" / generation / "document.md"
                    skipped = True
                else:
                    skipped = False
                    cache = folder / "extracted" / generation
                    connection.execute("INSERT OR IGNORE INTO jobs(generation) VALUES (?)", (generation,))
                    connection.commit()
                    try:
                        pages = self.extractor(data, self.config, cache, progress)
                        title = title.strip() or path.stem
                        chunks = chunks_for(pages, title, self.config)
                        if not chunks:
                            raise ValueError("検索できるテキストがありません。取り込みは完了していません。")
                        pages_count, chunks_count = len(pages), len(chunks)
                        markdown = cache / "document.md"
                        markdown.write_text("\n\n".join(f"## p. {page['page']}\n\n{page['text']}" for page in pages), encoding="utf-8")
                        table = self._table(folder, create=True)
                        completed = connection.execute("SELECT completed FROM jobs WHERE generation=?", (generation,)).fetchone()[0]
                        for start in range(completed, len(chunks), self.config.batch_size):
                            if progress:
                                progress("埋め込み", start, len(chunks))
                            batch = chunks[start:start + self.config.batch_size]
                            vectors = self.embedder.encode([chunk["embedding_text"] for chunk in batch])
                            if len(vectors) != len(batch):
                                raise ValueError("埋め込みの件数がチャンク数と一致しません。")
                            rows = []
                            for number, (chunk, vector) in enumerate(zip(batch, vectors), start):
                                vector = list(map(float, vector))
                                if len(vector) != self.config.dimension or not all(__import__("math").isfinite(v) for v in vector):
                                    raise ValueError("埋め込みの次元数または数値が不正です。")
                                rows.append({"chunk_id": f"{generation}:{number}", "doc_id": doc_id, "generation": generation,
                                             "page": chunk["page"], "section": chunk["section"], "title": title,
                                             "text": chunk["text"], "terms": self.tokenize(chunk["embedding_text"]), "vector": vector})
                            table.merge_insert("chunk_id").when_matched_update_all().when_not_matched_insert_all().execute(rows)
                            connection.execute("UPDATE jobs SET completed=?, error=NULL WHERE generation=?", (start + len(batch), generation))
                            connection.commit()
                            if progress:
                                progress("埋め込み", start + len(batch), len(chunks))
                    except Exception as error:
                        connection.execute("UPDATE jobs SET status='failed', error=? WHERE generation=?", (str(error), generation))
                        connection.commit()
                        raise
                self._maintain_indexes(folder,self._table(folder),progress)
                current = os.stat(path)
                if (stat_after.st_size, stat_after.st_mtime_ns) != (current.st_size, current.st_mtime_ns):
                    raise ValueError("処理中に PDF が変更されました。新しいジョブで取り込んでください。")
                if before_publish:
                    before_publish()
                # 新しい索引が揃うまでは旧版を検索対象として残す。
                reviews = self._mark_reviews(wiki, connection, old_id) if old_id else []
                connection.execute("""INSERT OR REPLACE INTO documents
                    (doc_id,generation,title,pages,chunks,fingerprint,ingested_at)
                    VALUES (?,?,?,?,?,?,?)""",
                                   (doc_id, generation, existing["title"] if existing else title,
                                    pages_count, chunks_count, self.config.fingerprint, time.time()))
                connection.execute("INSERT OR REPLACE INTO locations VALUES (?,?,?,?,?)",
                                   (location, doc_id, stat_after.st_size, stat_after.st_mtime_ns, time.time()))
                connection.execute("INSERT OR IGNORE INTO citations VALUES (?,?)", (doc_id, source_page))
                # 改名した同じ内容の資料は、存在しなくなった旧パスだけを台帳から外す。
                for row in connection.execute("SELECT path FROM locations WHERE doc_id=?", (doc_id,)).fetchall():
                    if not os.path.exists(row["path"]):
                        connection.execute("DELETE FROM locations WHERE path=?", (row["path"],))
                connection.execute("UPDATE jobs SET status='complete', error=NULL WHERE generation=?", (generation,))
                if old_id and not connection.execute("SELECT 1 FROM locations WHERE doc_id=?", (old_id,)).fetchone():
                    connection.execute("DELETE FROM documents WHERE doc_id=?", (old_id,))
                connection.commit()
                # コミット後の保守が失敗しても、新版は検索できる。旧行は active generation の条件で除外する。
                cleanup_error = None
                if old_id and not connection.execute("SELECT 1 FROM documents WHERE doc_id=?", (old_id,)).fetchone():
                    try:
                        self._table(folder).delete(f"doc_id = '{old_id}'")
                        self._maintain_indexes(folder,self._table(folder))
                        old_generation = hashlib.sha256((old_id + self.config.fingerprint).encode("ascii")).hexdigest()
                        old_cache = folder / "extracted" / old_generation
                        for cached in old_cache.glob("*.json"):
                            cached.unlink()
                        (old_cache / "document.md").unlink(missing_ok=True)
                    except Exception as error:
                        cleanup_error = str(error)
                return {"status": "unchanged" if skipped else "indexed", "document_id": doc_id,
                        "raw": path.as_posix(), "sha256": doc_id, "pages": pages_count, "chunks": chunks_count,
                        "extracted_markdown": str(markdown), "source_page": source_page,
                        "review_pages": reviews, "cleanup_error": cleanup_error,
                        "indexes":json.loads((folder/'index-state.json').read_text(encoding='utf-8'))}

    def search(self, wiki_root: str, query: str, limit: int = 10) -> dict:
        if not query.strip() or not 1 <= limit <= 50:
            raise ValueError("空でない質問と、1〜50 の limit を指定してください。")
        folder = self.config.workspace(wiki_root)
        empty={"results":[],"model":self.config.model,"reranker":self.config.reranker,
               "abstained":self.config.min_rerank_score is not None}
        if not (folder / "catalog.sqlite").exists():
            return empty
        with FileLock(str(folder / "write.lock"), timeout=1800):
            self._check_policy(folder)
            with self._catalog(folder) as connection:
                documents = {row["doc_id"]: dict(row) for row in connection.execute("SELECT * FROM documents")}
                if not documents:
                    return empty
                active = "generation IN (" + ",".join("'" + row["generation"] + "'" for row in documents.values()) + ")"
                table = self._table(folder)
                vector = self.embedder.encode([query], query=True)[0].tolist()
                candidates = max(limit, min(200, limit * 10))
                columns = ["chunk_id", "doc_id", "page", "section", "title", "text"]
                dense_query = table.search(vector).distance_type("cosine").where(active, prefilter=True).select(columns)
                if any("vector" in index.columns for index in table.list_indices()):
                    dense_query = dense_query.nprobes(self.config.search_nprobes).refine_factor(self.config.search_refine_factor)
                dense = dense_query.limit(candidates).to_list()
                terms = self.tokenize(query)
                lexical = table.search(terms, query_type="fts", fts_columns="terms").where(active, prefilter=True).select(columns).limit(candidates).to_list() if terms else []
                scores, rows = {}, {}
                for ranking in (dense, lexical):
                    for rank, row in enumerate(ranking, 1):
                        key = row["chunk_id"]
                        rows[key] = row
                        scores[key] = scores.get(key, 0) + 1 / (60 + rank)
                ranked = sorted(scores, key=lambda k: (-scores[k], k))
                reranked_scores = {}
                if self.config.reranker != "off" and ranked:
                    top = ranked[:self.config.rerank_limit]
                    values = self.reranker.score([(query, f"{rows[key]['title']}\n{rows[key]['section']}\n{rows[key]['text']}") for key in top])
                    if len(values) != len(top) or not all(__import__("math").isfinite(value) for value in values):
                        raise ValueError("リランカーの件数または数値が不正です。")
                    reranked_scores = dict(zip(top, values))
                    ranked = sorted(top, key=lambda key: (-reranked_scores[key], key)) + ranked[len(top):]
                    if self.config.min_rerank_score is not None:
                        ranked = [key for key in ranked if key in reranked_scores and reranked_scores[key] >= self.config.min_rerank_score]
                results = []
                for key in ranked[:limit]:
                    row = rows[key]
                    found = self._location(connection, row["doc_id"])
                    sources = [item[0] for item in connection.execute("SELECT source_page FROM citations WHERE doc_id=?", (row["doc_id"],))]
                    results.append({**{k: v for k, v in row.items() if not k.startswith("_")}, "score": scores[key],
                                    "raw": Path(found).as_posix() if found else None, "sha256": row["doc_id"],
                                    "source_pages": sources,
                                    "rerank_score": reranked_scores.get(key)})
                return {"results": results, "model": self.config.model, "reranker":self.config.reranker,
                        "abstained":not results and self.config.min_rerank_score is not None}

    def status(self, wiki_root: str) -> dict:
        folder = self.config.workspace(wiki_root)
        if not (folder / "catalog.sqlite").exists():
            return {"documents": 0, "chunks": 0, "jobs": [], "data_dir": str(folder)}
        with self._catalog(folder) as connection:
            counts = connection.execute("SELECT COUNT(*),COALESCE(SUM(chunks),0) FROM documents").fetchone()
            jobs = [dict(row) for row in connection.execute("SELECT generation,completed,status,error FROM jobs WHERE status != 'complete'")]
            index_state=folder/'index-state.json'
            return {"documents": counts[0], "chunks": counts[1], "jobs": jobs, "data_dir": str(folder),
                    "indexes":json.loads(index_state.read_text(encoding='utf-8')) if index_state.exists() else None}
