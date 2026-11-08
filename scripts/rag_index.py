"""wiki の PDF 取り込みを起点とするローカル索引。回答生成は行わない。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import time
from datetime import timedelta
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

from rag_config import MODELS, Config, write_json
from rag_pdf import chunks_for, extract_pages


def file_id(info: os.stat_result) -> tuple[str, str] | None:
    """ファイルの識別子（st_dev と st_ino）を文字列で返す。Windows の st_ino は 64 ビットに収まらないことがある。
    識別子を返さないファイルシステム（st_ino が 0）では None を返す。"""
    return (str(info.st_dev), str(info.st_ino)) if info.st_ino else None


def _same_file(first: os.stat_result, second: os.stat_result) -> bool:
    return (first.st_dev, first.st_ino, first.st_size, first.st_mtime_ns) == (
        second.st_dev, second.st_ino, second.st_size, second.st_mtime_ns)


def read_pdf(path: Path, expected: tuple | None = None) -> tuple[bytes, os.stat_result]:
    """確かめたファイルを 1 回だけ開いて読み、読んだバイト列が PDF であることを確かめる（ADR-0013）。

    expected は登録したときのファイルの (st_dev, st_ino, st_size, st_mtime_ns)。開いたファイルがこれと
    違えば読まない。登録のあとで親のフォルダーが差し替えられても、別のファイルを読まないため。
    """
    before = os.lstat(path)
    try:
        handle = open(path, "rb")
    except OSError as error:
        raise ValueError(f"PDF を開けません（{error.strerror or error}）。") from error
    with handle:
        opened = os.fstat(handle.fileno())
        if not _same_file(before, opened):
            raise ValueError("確かめた PDF と開いたファイルが違います。もう一度取り込んでください。")
        if expected is not None:
            dev, ino, size, mtime_ns = expected
            if (opened.st_size, opened.st_mtime_ns) != (size, mtime_ns) or (
                    ino not in (None, "", "0", 0) and file_id(opened) != (str(dev), str(ino))):
                raise ValueError("登録した PDF と開いたファイルが違います。新しいジョブで取り込んでください。")
        try:
            data = handle.read()
        except OSError as error:
            # OneDrive などのオンラインだけのファイルは、同期のアプリが動いていないと読めない（Errno 22 になる）。
            raise ValueError(f"PDF を読めません（{error.strerror or error}）。OneDrive などのオンラインだけの"
                             "ファイルなら、同期のアプリを起動してから、もう一度取り込んでください。") from error
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
                confirmed_at REAL NOT NULL, dev TEXT, ino TEXT);
            CREATE INDEX IF NOT EXISTS locations_file ON locations(dev, ino);
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

    def _write_policy(self, folder: Path) -> None:
        write_json(folder / "policy.json", {"fingerprint": self.config.fingerprint, "model": self.config.model,
                                            "revision": self.config.revision, "dimension": self.config.dimension})

    def _stale_policy(self, folder: Path) -> dict | None:
        """索引の方針が今の設定と違えば、記録した方針を返す。同じか、まだ索引がなければ None を返す。"""
        path = folder / "policy.json"
        if not path.exists():
            return None
        previous = json.loads(path.read_text(encoding="utf-8"))
        return previous if previous["fingerprint"] != self.config.fingerprint else None

    def _check_policy(self, folder: Path) -> None:
        """検索と索引の保守の前に呼ぶ。方針が変わった索引は、取り込み直すまで使わない（ADR-0014）。"""
        if self._stale_policy(folder):
            raise ValueError("埋め込みのモデル、分割、OCR の設定が変わったので、PDF を取り込み直すまで検索できません。"
                             "取り込み直す PDF は status の reingest で確かめてください。")
        if not (folder / "policy.json").exists():
            self._write_policy(folder)

    @staticmethod
    def _previous_documents(folder: Path) -> dict:
        path = folder / "previous-index.json"
        if not path.exists():
            return {}
        return {item["sha256"]: item for item in json.loads(path.read_text(encoding="utf-8"))["documents"]}

    @staticmethod
    def _catalog_documents(connection) -> dict:
        documents = {}
        for row in connection.execute("SELECT doc_id,title FROM documents ORDER BY ingested_at"):
            paths = [Path(item[0]).as_posix() for item in connection.execute(
                "SELECT path FROM locations WHERE doc_id=? ORDER BY confirmed_at DESC", (row["doc_id"],))]
            pages = [item[0] for item in connection.execute(
                "SELECT source_page FROM citations WHERE doc_id=? ORDER BY source_page", (row["doc_id"],))]
            documents[row["doc_id"]] = {"sha256": row["doc_id"], "title": row["title"], "paths": paths, "source_pages": pages}
        return documents

    def _reset_index(self, folder: Path, connection, stale: dict) -> None:
        """方針が変わった索引を空にする。先に取り込み直す文書の一覧を残し、途中で止まっても、
        次の取り込みでもう一度空にすれば同じ状態になるようにする（ADR-0014）。"""
        documents = self._merge_documents(self._previous_documents(folder), self._catalog_documents(connection))
        # 空にしたあとも 0.4.0 の抽出の結果を探せるよう、これまでの方針も残す。
        policies = self._previous_policies(folder)
        policy = {key: stale.get(key) for key in ("fingerprint", "model", "revision")}
        if policy not in policies:
            policies.append(policy)
        write_json(folder / "previous-index.json", {"fingerprint": stale["fingerprint"], "model": stale.get("model"),
                                                    "policies": policies, "documents": list(documents.values())})
        for table in ("documents", "locations", "citations", "jobs"):
            connection.execute(f"DELETE FROM {table}")
        connection.commit()
        import lancedb
        database = lancedb.connect(str(folder / "lance"), read_consistency_interval=timedelta(0))
        try:
            database.drop_table("chunks")
        except (FileNotFoundError, ValueError):
            pass
        (folder / "index-state.json").unlink(missing_ok=True)
        self._write_policy(folder)

    @staticmethod
    def _merge_documents(previous: dict, current: dict) -> dict:
        """取り込み直しの一覧と台帳の文書を合わせる。同じ文書は、場所とソースのページの和をとる。
        取り込み直しの途中でもう一度設定を変えても、まだ取り込み直していない場所やページを落とさない。"""
        documents = dict(previous)
        for doc_id, item in current.items():
            if doc_id in documents:
                before = documents[doc_id]
                item = {**item, "paths": list(dict.fromkeys(item["paths"] + before["paths"])),
                        "source_pages": sorted(set(item["source_pages"]) | set(before["source_pages"]))}
            documents[doc_id] = item
        return documents

    @staticmethod
    def _gone(place: str) -> bool:
        """その場所の PDF が消えたと言えるなら True。フォルダーはあるのにファイルだけがないときに限る。
        ネットワークドライブが切れているときなど、フォルダーごと見えないときは、消えたとはみなさない。"""
        try:
            return not os.path.exists(place) and os.path.isdir(os.path.dirname(place))
        except OSError:
            return False

    def _reingest(self, folder: Path, connection) -> list[dict]:
        """取り込み直すべき文書。方針が変わった索引の文書と、空にする前に残した一覧のうち、まだ取り込み直していない場所。
        同じ内容を 2 か所で登録していれば、片方を取り込み直しても、もう片方が残る。台帳に載っている場所と、消えたと言える
        場所（_gone）は外す。場所が残らない文書は外す。"""
        pending = self._previous_documents(folder)
        current = self._catalog_documents(connection)
        if self._stale_policy(folder):
            return list(self._merge_documents(pending, current).values())
        placed = {place for item in current.values() for place in item["paths"]}
        result = []
        for doc_id, item in pending.items():
            paths = [place for place in item["paths"] if place not in placed and not self._gone(place)]
            if paths:
                result.append({**item, "paths": paths})
        return result

    def _extraction_lock(self, doc_id: str, timeout: float = 1800) -> FileLock:
        self.config.extraction_root.mkdir(parents=True, exist_ok=True)
        return FileLock(str(self.config.extraction_root / f"{doc_id}.lock"), timeout=timeout)

    @staticmethod
    def _write_markdown(markdown: Path, pages: list[dict]) -> None:
        """document.md は題名を含まないので、wiki によらない。途中で止まっても半端なファイルを残さない。"""
        temporary = markdown.with_name(markdown.name + ".tmp")
        temporary.write_text("\n\n".join(f"## p. {page['page']}\n\n{page['text']}" for page in pages), encoding="utf-8")
        os.replace(temporary, markdown)

    @staticmethod
    def _previous_policies(folder: Path) -> list[dict]:
        path = folder / "previous-index.json"
        if not path.exists():
            return []
        return list(json.loads(path.read_text(encoding="utf-8")).get("policies", []))

    def _legacy_generations(self, folder: Path, doc_id: str, stale: dict | None) -> list[str]:
        """0.4.0 が抽出の結果を置いた generation（0.4.0 の式 sha256(doc_id + 指紋)）の候補。今の指紋のものと、
        前の方針（今の索引の方針と、空にした索引の方針）のうち埋め込みのモデルだけが違うもの。どれも、OCR の設定が
        今と同じであることが指紋から分かる。抽出の版は指紋に入らないので、移すかどうかは呼び出し側が版で決める。"""
        fingerprints = [self.config.fingerprint]
        for policy in ([stale] if stale else []) + self._previous_policies(folder):
            if policy.get("model") not in MODELS or policy.get("fingerprint") in fingerprints:
                continue
            try:
                previous = Config({**self.config.values, "data_dir": str(self.config.data_dir),
                                   "embedding_model": policy["model"], "embedding_revision": policy.get("revision")})
            except ValueError:
                continue
            if previous.fingerprint == policy["fingerprint"]:
                fingerprints.append(policy["fingerprint"])
        return [hashlib.sha256((doc_id + value).encode("ascii")).hexdigest() for value in fingerprints]

    @staticmethod
    def _remove_legacy_extraction(folder: Path, generations: list[str]) -> None:
        """0.4.0 が wiki の作業領域に置いたまま移していない抽出の結果を消す。"""
        for generation in generations:
            shutil.rmtree(folder / "extracted" / generation, ignore_errors=True)
        try:
            (folder / "extracted").rmdir()
        except OSError:
            pass

    def _adopt_legacy_extraction(self, folder: Path, generations: list[str], cache: Path) -> None:
        """0.4.0 が wiki の作業領域に置いた抽出の結果を、新しい置き場所に移す。OCR の設定が同じものだけを移す。
        0.4.0 の結果は extract-v1 の抽出なので、抽出の版を上げたあとは移さない（古い本文を新しい版の結果にしない）。
        移す途中で止まっても、次の取り込みで、まだ移していないファイルを移す。"""
        if not self.config.adopts_legacy_extraction:
            return
        legacy = next((folder / "extracted" / generation for generation in generations
                       if (folder / "extracted" / generation).is_dir()), None)
        if legacy is None:
            return
        cache.mkdir(parents=True, exist_ok=True)
        for item in legacy.iterdir():
            if item.is_file() and (item.name == "document.md" or re.fullmatch(r"page-\d{6}\.json", item.name)) \
                    and not (cache / item.name).exists():
                os.replace(item, cache / item.name)
        shutil.rmtree(legacy, ignore_errors=True)
        try:
            (folder / "extracted").rmdir()
        except OSError:
            pass

    def _extraction_in_use(self, doc_id: str) -> bool:
        """同じ data_dir のどこかの wiki が、その文書を索引に持つか、取り込み直しを待っていれば True。"""
        for workspace in (self.config.data_dir / "wikis-v2").glob("*"):
            catalog = workspace / "catalog.sqlite"
            if not catalog.exists():
                if doc_id in self._previous_documents(workspace):
                    return True
                continue
            try:
                with closing(sqlite3.connect(catalog, timeout=15)) as connection:
                    connection.row_factory = sqlite3.Row
                    if connection.execute("SELECT 1 FROM documents WHERE doc_id=?", (doc_id,)).fetchone():
                        return True
                    # 取り込み直しの一覧は、まだ取り込み直していない文書だけを使用中とみなす。
                    if any(item["sha256"] == doc_id for item in self._reingest(workspace, connection)):
                        return True
            except (sqlite3.Error, OSError, ValueError, KeyError):
                return True  # 読めない台帳があれば、消さずに残す
        return False

    def _prune_previous(self, folder: Path, connection) -> None:
        """取り込み直しの一覧から、取り込み直した文書と置き換えられた文書を外し、外した文書の抽出の結果を後始末する。"""
        previous = self._previous_documents(folder)
        if not previous:
            return
        pending = self._reingest(folder, connection)
        # 一覧を消す前に、0.4.0 の置き場所を探すための方針を読んでおく。
        legacy = {doc_id: self._legacy_generations(folder, doc_id, None) for doc_id in previous}
        if pending:
            data = json.loads((folder / "previous-index.json").read_text(encoding="utf-8"))
            write_json(folder / "previous-index.json", {**data, "documents": pending})
        else:
            (folder / "previous-index.json").unlink(missing_ok=True)
        current = {row[0] for row in connection.execute("SELECT doc_id FROM documents")}
        keep = {item["sha256"] for item in pending} | current
        for doc_id in previous:
            if doc_id not in keep:
                self._remove_extraction(doc_id)
                self._remove_legacy_extraction(folder, legacy[doc_id])

    def _remove_extraction(self, doc_id: str) -> None:
        """どの wiki も使わなくなった文書の抽出の結果を、抽出の設定によらずすべて消す。"""
        try:
            with self._extraction_lock(doc_id, timeout=0):
                if not self._extraction_in_use(doc_id):
                    shutil.rmtree(self.config.extraction_root / doc_id)
        except Timeout:
            pass  # ほかの取り込みが使っている。消さずに残す
        except FileNotFoundError:
            pass

    @staticmethod
    def _location(connection, doc_id: str) -> str | None:
        """文書の今の場所を返す。最後に確かめた順に見て、記録した大きさと更新時刻が今と合う場所を選ぶ。

        合う場所がなければ None を返す。中身が変わったかもしれない場所を、古い版の出典として返さない。
        """
        rows = connection.execute("SELECT path,size,mtime_ns FROM locations WHERE doc_id=? ORDER BY confirmed_at DESC",
                                  (doc_id,)).fetchall()
        for row in rows:
            try:
                current = os.stat(row["path"])
            except OSError:
                continue
            if (current.st_size, current.st_mtime_ns) == (row["size"], row["mtime_ns"]):
                return row["path"]
        return None

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

    def _mark_reviews(self, wiki: Path, connection, doc_id: str, cited=()) -> list[str]:
        import yaml

        direct = {row[0] for row in connection.execute("SELECT source_page FROM citations WHERE doc_id=?", (doc_id,))}
        direct.update(cited)
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
                   before_publish=None, expected: tuple | None = None) -> dict:
        wiki = Path(wiki_root).expanduser().resolve()
        folder = self.config.workspace(str(wiki))
        path = self.config.pdf_path(source)
        location = str(path)
        source_page = self._source_page(wiki, source_page)
        folder.mkdir(parents=True, exist_ok=True)
        with FileLock(str(folder / "write.lock"), timeout=1800), ExitStack() as held:
            # 方針が変わった索引は、抽出が済んでから空にする。それまでは台帳の文書を今の方針のものとみなさない。
            stale = self._stale_policy(folder)
            if not stale and not (folder / "policy.json").exists():
                self._write_policy(folder)
            index_reset = False
            with self._catalog(folder) as connection:
                if progress:
                    progress("PDF の読み込み", 0, 1)
                # 遅い原資料を一度だけ順次読み込み、ハッシュと PDFium で共用する。
                data, stat_after = read_pdf(path, expected)
                if progress:
                    progress("PDF の読み込み", 1, 1)
                doc_id = hashlib.sha256(data).hexdigest()
                generation = self.config.generation(doc_id)
                # 抽出の結果は wiki のあいだで共有する。使っているあいだは、ほかの取り込みの後始末で消させない。
                held.enter_context(self._extraction_lock(doc_id))
                cache = self.config.extraction_dir(doc_id)
                self._adopt_legacy_extraction(folder, self._legacy_generations(folder, doc_id, stale), cache)
                markdown = cache / "document.md"
                existing = None if stale else connection.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
                if existing:
                    pages_count, chunks_count = existing["pages"], existing["chunks"]
                    # 索引の行は、記録した generation で入っている（抽出の版を上げる前の式のこともある）。
                    generation = existing["generation"]
                    skipped = True
                    if not markdown.exists():
                        # 索引にはあるが、今の抽出の設定の結果がない（抽出の版を上げた、結果を消したなど）。
                        # 索引は作り直さず、返す本文のために抽出だけをやり直す。
                        self._write_markdown(markdown, self.extractor(data, self.config, cache, progress))
                else:
                    skipped = False
                    if not stale:
                        connection.execute("INSERT OR IGNORE INTO jobs(generation) VALUES (?)", (generation,))
                        connection.commit()
                    try:
                        pages = self.extractor(data, self.config, cache, progress)
                        title = title.strip() or path.stem
                        chunks = chunks_for(pages, title, self.config)
                        if not chunks:
                            raise ValueError("検索できるテキストがありません。取り込みは完了していません。")
                        pages_count, chunks_count = len(pages), len(chunks)
                        if not markdown.exists():
                            self._write_markdown(markdown, pages)
                        # 方針が変わった索引は、最初のバッチの埋め込みが済んでから空にする。モデルの取得や読み込みに
                        # 失敗しても、前の索引を壊さない。設定を戻せば、前の索引をそのまま使える。
                        table = None if stale else self._table(folder, create=True)
                        completed = 0 if stale else connection.execute(
                            "SELECT completed FROM jobs WHERE generation=?", (generation,)).fetchone()[0]
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
                            if stale:
                                self._reset_index(folder, connection, stale)
                                stale, index_reset = None, True
                                connection.execute("INSERT OR IGNORE INTO jobs(generation) VALUES (?)", (generation,))
                                connection.commit()
                            if table is None:
                                table = self._table(folder, create=True)
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
                # この場所と、同じファイルの別名（ハードリンクなど）の場所が前に指していた文書は、古い版になる。
                # 別名は公開の直前に調べる。取り込みの途中で別名が差し替えられても、残っている古い版を消さないため。
                same_file = file_id(stat_after)
                previous = connection.execute(
                    "SELECT path,doc_id FROM locations WHERE path=? OR (? IS NOT NULL AND dev=? AND ino=?)",
                    (location, same_file and 1, *(same_file or (None, None)))).fetchall()
                aliases, affected, vanished = [], [], []
                for row in previous:
                    if row["path"] == location:
                        affected.append(row)
                        continue
                    # 記録した識別子だけでは決めない。今も同じファイルを指しているときだけ別名とみなす。
                    try:
                        current = os.stat(row["path"])
                    except FileNotFoundError:
                        vanished.append(row)  # もうない別名の行は外す
                        continue
                    except OSError:
                        continue
                    if file_id(current) == same_file and (current.st_size, current.st_mtime_ns) == (
                            stat_after.st_size, stat_after.st_mtime_ns):
                        aliases.append(row["path"])
                        affected.append(row)
                old_ids = sorted({row["doc_id"] for row in affected + vanished if row["doc_id"] != doc_id})
                for row in vanished:
                    connection.execute("DELETE FROM locations WHERE path=?", (row["path"],))
                # 新しい索引が揃うまでは旧版を検索対象として残す。
                reviews = {page for old_id in old_ids for page in self._mark_reviews(wiki, connection, old_id)}
                # 索引を空にしたあとは、この場所の前の版を台帳で引けない。取り込み直しの一覧から引く。
                for item in self._previous_documents(folder).values():
                    if item["sha256"] != doc_id and path.as_posix() in item["paths"]:
                        reviews.update(self._mark_reviews(wiki, connection, item["sha256"], item["source_pages"]))
                reviews = sorted(reviews)
                connection.execute("""INSERT OR REPLACE INTO documents
                    (doc_id,generation,title,pages,chunks,fingerprint,ingested_at)
                    VALUES (?,?,?,?,?,?,?)""",
                                   (doc_id, generation, existing["title"] if existing else title,
                                    pages_count, chunks_count, self.config.fingerprint, time.time()))
                now = time.time()
                for place in [location, *aliases]:
                    connection.execute("INSERT OR REPLACE INTO locations VALUES (?,?,?,?,?,?,?)",
                                       (place, doc_id, stat_after.st_size, stat_after.st_mtime_ns,
                                        now if place == location else now - 1e-6, *(same_file or (None, None))))
                connection.execute("INSERT OR IGNORE INTO citations VALUES (?,?)", (doc_id, source_page))
                # 索引を空にする前に、この文書を出典にしていたページも戻す。版が変わったときの要再確認の印に使う。
                restored = self._previous_documents(folder).get(doc_id)
                for page in restored["source_pages"] if restored else []:
                    connection.execute("INSERT OR IGNORE INTO citations VALUES (?,?)", (doc_id, page))
                # 改名した同じ内容の資料は、存在しなくなった旧パスだけを台帳から外す。
                for row in connection.execute("SELECT path FROM locations WHERE doc_id=?", (doc_id,)).fetchall():
                    if not os.path.exists(row["path"]):
                        connection.execute("DELETE FROM locations WHERE path=?", (row["path"],))
                connection.execute("UPDATE jobs SET status='complete', error=NULL WHERE generation=?", (generation,))
                removed = []
                old_rows = {old_id: connection.execute("SELECT generation FROM documents WHERE doc_id=?", (old_id,)).fetchall()
                            for old_id in old_ids}
                for old_id in old_ids:
                    if not connection.execute("SELECT 1 FROM locations WHERE doc_id=?", (old_id,)).fetchone():
                        connection.execute("DELETE FROM documents WHERE doc_id=?", (old_id,))
                        # 内部の途中経過も消す。同じ内容に戻したとき、埋め込みを最初からやり直すため。
                        # 記録した generation は、抽出の版を上げる前の式のこともあるので、今の式の値と両方を消す。
                        old_generations = {self.config.generation(old_id), *(row[0] for row in old_rows.get(old_id, []))}
                        connection.executemany("DELETE FROM jobs WHERE generation=?", [(value,) for value in old_generations])
                        removed.append((old_id, self._legacy_generations(folder, old_id, None)))
                connection.commit()
                # コミット後の保守が失敗しても、新版は検索できる。旧行は active generation の条件で除外する。
                cleanup_error = None
                try:
                    # 取り込み直しの一覧から、取り込み直した文書と置き換えられた文書を外す。すべて外れたら一覧を消す。
                    self._prune_previous(folder, connection)
                except Exception as error:
                    cleanup_error = str(error)
                for old_id, legacy_generations in removed:
                    try:
                        self._table(folder).delete(f"doc_id = '{old_id}'")
                        self._maintain_indexes(folder,self._table(folder))
                    except Exception as error:
                        cleanup_error = str(error)
                    try:
                        # 抽出の結果は、同じ data_dir のどの wiki も使わなくなったときだけ消す（ADR-0014）。
                        self._remove_extraction(old_id)
                        # 0.4.0 がこの wiki に置いたまま移していない結果も消す。
                        self._remove_legacy_extraction(folder, legacy_generations)
                    except Exception as error:
                        cleanup_error = str(error)
                return {"status": "unchanged" if skipped else "indexed", "document_id": doc_id,
                        "raw": path.as_posix(), "sha256": doc_id, "pages": pages_count, "chunks": chunks_count,
                        "extracted_markdown": str(markdown), "source_page": source_page,
                        "review_pages": reviews, "cleanup_error": cleanup_error, "index_reset": index_reset,
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
            return {"documents": 0, "chunks": 0, "jobs": [], "data_dir": str(folder),
                    "reingest_required": False, "reingest": []}
        with self._catalog(folder) as connection:
            counts = connection.execute("SELECT COUNT(*),COALESCE(SUM(chunks),0) FROM documents").fetchone()
            jobs = [dict(row) for row in connection.execute("SELECT generation,completed,status,error FROM jobs WHERE status != 'complete'")]
            index_state=folder/'index-state.json'
            # 設定が変わった索引の文書と、空にしたあとまだ取り込み直していない文書（ADR-0014）。
            reingest = self._reingest(folder, connection)
            return {"documents": counts[0], "chunks": counts[1], "jobs": jobs, "data_dir": str(folder),
                    "indexes":json.loads(index_state.read_text(encoding='utf-8')) if index_state.exists() else None,
                    "reingest_required": bool(reingest), "reingest": reingest}
