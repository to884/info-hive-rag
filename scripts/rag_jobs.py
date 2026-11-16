"""PDF の登録と進捗を保存する共通コア。モデルはワーカーだけが読み込む。"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
from contextlib import closing, contextmanager
from pathlib import Path

from rag_config import Config


class JobCancelled(Exception):
    """キャンセル済み、または別の実行者に移ったジョブ。"""


class Jobs:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # 0.3.0 の jobs.sqlite は開かない（ADR-0013）。
        self.path = self.data_dir / "jobs-v2.sqlite"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS ingest_jobs (
                    id TEXT PRIMARY KEY, wiki TEXT NOT NULL, source TEXT NOT NULL,
                    source_page TEXT NOT NULL, title TEXT NOT NULL, size INTEGER, mtime_ns INTEGER,
                    config TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued',
                    phase TEXT NOT NULL DEFAULT '受付', completed INTEGER NOT NULL DEFAULT 0,
                    total INTEGER NOT NULL DEFAULT 0, attempts INTEGER NOT NULL DEFAULT 0,
                    owner TEXT, error TEXT, result TEXT, created_at REAL, updated_at REAL,
                    publication_started INTEGER NOT NULL DEFAULT 0, request_key TEXT NOT NULL,
                    dev TEXT, ino TEXT);
                CREATE INDEX IF NOT EXISTS ingest_jobs_request ON ingest_jobs(request_key, status);
                CREATE TABLE IF NOT EXISTS search_requests (
                    id TEXT PRIMARY KEY, config TEXT NOT NULL, wiki TEXT NOT NULL,
                    query TEXT NOT NULL, result_limit INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued', result TEXT, error TEXT,
                    updated_at REAL NOT NULL);
            """)
            # rag 0.6.0 で、確認を当てる作り直しのジョブ（kind='review'）を足した（ADR-0025）。前の版の台帳には列を足す。
            columns = {row[1] for row in db.execute("PRAGMA table_info(ingest_jobs)")}
            for name, declaration in (("kind", "TEXT NOT NULL DEFAULT 'ingest'"), ("doc_id", "TEXT"),
                                      ("rerun", "INTEGER NOT NULL DEFAULT 0")):
                if name not in columns:
                    db.execute(f"ALTER TABLE ingest_jobs ADD COLUMN {name} {declaration}")

    @contextmanager
    def connect(self):
        # sqlite3.Connection の with だけでは close されないため、明示的に閉じる。
        with closing(sqlite3.connect(self.path, timeout=15)) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA busy_timeout=15000")
            with db:
                yield db

    def register(self, config: Config, wiki_root: str, source: str, source_page: str, title: str = "") -> dict:
        wiki = Path(wiki_root).expanduser().resolve()
        config.workspace(str(wiki))
        path = config.pdf_path(source)
        page = config.source_page(wiki, source_page)
        stat = os.stat(path)
        key = [os.path.normcase(str(wiki)), str(path), stat.st_size, stat.st_mtime_ns,
               config.fingerprint, page, title.strip()]
        request_key = hashlib.sha256(json.dumps(key, ensure_ascii=False).encode("utf-8")).hexdigest()
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            # 同じ登録が受け付け済みか処理中なら、そのジョブを返す。完了、失敗、キャンセルしかなければ
            # 新しいジョブにする。大きさと更新時刻を変えずに中身を差し替えた PDF も読み直すため（ADR-0013）。
            row = db.execute("""SELECT id FROM ingest_jobs WHERE request_key=?
                AND status IN ('queued','running','paused') ORDER BY created_at DESC LIMIT 1""", (request_key,)).fetchone()
            if row:
                identity = row[0]
            else:
                identity = uuid.uuid4().hex
                db.execute("""INSERT INTO ingest_jobs
                    (id,wiki,source,source_page,title,size,mtime_ns,config,created_at,updated_at,request_key,dev,ino)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (identity, str(wiki), config.pdf_source(source), page, title.strip(), stat.st_size, stat.st_mtime_ns,
                     json.dumps(config.snapshot(), ensure_ascii=False), now, now, request_key, str(stat.st_dev), str(stat.st_ino)))
        return self.status(str(wiki), identity)

    def register_rebuild(self, config: Config, wiki_root: str, doc_id: str) -> dict:
        """確認の結果を当てる作り直しのジョブを登録する。wiki と PDF ごとに 1 つだけ持つ（ADR-0025）。

        取り込みのジョブの重複の除き方（request_key）は使わない。同じ wiki と PDF の作り直しが待っていれば、それを返す
        （始めるときに台帳の最新の版を読む）。処理中なら rerun の印を付けて返し、終わったときにもう一度待ちに戻す
        （処理中に受け付けた確認を取りこぼさない）。
        """
        wiki = Path(wiki_root).expanduser().resolve()
        config.workspace(str(wiki))
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("""SELECT id,status FROM ingest_jobs WHERE kind='review' AND wiki=? AND doc_id=?
                AND status IN ('queued','running','paused') ORDER BY created_at DESC LIMIT 1""", (str(wiki), doc_id)).fetchone()
            if row:
                identity = row["id"]
                if row["status"] == "running":
                    db.execute("UPDATE ingest_jobs SET rerun=1,updated_at=? WHERE id=?", (now, identity))
            else:
                identity = uuid.uuid4().hex
                db.execute("""INSERT INTO ingest_jobs
                    (id,wiki,source,source_page,title,size,mtime_ns,config,created_at,updated_at,request_key,kind,doc_id,phase)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (identity, str(wiki), "", "", "", 0, 0, json.dumps(config.snapshot(), ensure_ascii=False), now, now,
                     "review:" + identity, "review", doc_id, "確認の結果を当てる作り直しを受付"))
        return self.status(str(wiki), identity)

    def status(self, wiki_root: str, job_id: str, page_report_from: int = 0, threshold: float | None = None) -> dict:
        """ジョブの状態を返す。結果の page_report の一覧は、page_report_from の位置から 200 個の範囲を返す。

        page_report の確認の状態（確信度の低いページ、確認待ち、反映待ちなど）は、返すときに、その wiki の台帳と
        公開した版から作る。threshold を省くと、ジョブを登録したときの設定の閾値を使う。
        """
        wiki = str(Path(wiki_root).expanduser().resolve())
        with self.connect() as db:
            row = db.execute("SELECT * FROM ingest_jobs WHERE id=? AND wiki=?", (job_id, wiki)).fetchone()
        if row is None:
            raise ValueError("この wiki に指定のジョブがありません。")
        result = dict(row)
        job_config = json.loads(result["config"])
        result.pop("config")
        result.pop("rerun", None)
        result.pop("owner")
        result.pop("request_key")
        result.pop("dev")
        result.pop("ino")
        result["job_id"] = result.pop("id")
        result["result"] = json.loads(result["result"]) if result["result"] else None
        if result["result"] and result["result"].get("page_report"):
            from rag_pdf import render_page_report
            report = result["result"]["page_report"]
            if threshold is None:
                threshold = Config(job_config).ocr_confidence_threshold
            report = self._with_review(report, job_config, wiki, result["result"].get("sha256"), threshold)
            result["result"]["page_report"] = render_page_report(report, page_report_from)
        return result

    @staticmethod
    def _with_review(report: dict, job_config: dict, wiki: str, doc_id: str | None, threshold: float) -> dict:
        """page_report に、確信度の低いページと確認の状態を足す（ADR-0025 の「状態の表示」）。"""
        from rag_config import OCR_CONFIDENCE_THRESHOLD_CALIBRATED
        from rag_pdf_layer import page_ranges
        from rag_review import Ledger, load_summary, published_documents, published_folder, review_lists
        confidence = report.get("confidence") or {}
        # 確認の状態は、ジョブを保存したときの記録ではなく、その wiki で今公開している版の要約から作る
        # （OCR をやり直して本文が変わったあとも、公開している版の状態を返す）。文書がその wiki の索引になければ、
        # ジョブの記録の確信度だけを使い、確認の状態は返さない
        summary = {int(page): entry for page, entry in confidence.get("pages", {}).items()}
        published_version, ledger_version, ledger_decisions, published_decisions, published = 0, 0, {}, {}, None
        if doc_id:
            folder = Config(job_config).workspace(wiki)
            ledger = Ledger(folder)
            ledger_version = ledger.version(doc_id)
            published = published_documents(folder).get(doc_id)
            if published:
                published_version = published["review_version"]
                summary = load_summary(folder, doc_id, published["generation"]) or summary
                ledger_decisions = ledger.decisions(doc_id)
                published_decisions = ledger.decisions(doc_id, upto=published_version)
        lists = dict(report["lists"])
        for name, numbers in review_lists(summary, ledger_decisions, published_decisions, threshold).items():
            lists[name] = {"count": len(numbers), "ranges": page_ranges(numbers)}
        markdown = None
        if published:
            reviewed = published_folder(folder, doc_id, published["generation"]) / "document.md"
            markdown = str(reviewed) if published_version and reviewed.exists() else None
        return {**report, "lists": lists,
                "review": {"ledger_version": ledger_version, "published_version": published_version,
                           "indexed": bool(published),
                           # 確認を当てた本文（公開している版）。確認の結果がなければ None（ジョブの extracted_markdown を読む）
                           "extracted_markdown": markdown},
                "confidence": {"revision": confidence.get("revision"), "threshold": threshold,
                               "threshold_calibrated": OCR_CONFIDENCE_THRESHOLD_CALIBRATED}}

    def completed_document(self, wiki_root: str, job_id: str) -> str:
        """完了した取り込みのジョブの PDF の SHA-256 を返す。失敗したジョブと作り直しのジョブは受け付けない。"""
        wiki = str(Path(wiki_root).expanduser().resolve())
        with self.connect() as db:
            row = db.execute("SELECT status,result,kind FROM ingest_jobs WHERE id=? AND wiki=?", (job_id, wiki)).fetchone()
        if row is None:
            raise ValueError("この wiki に指定のジョブがありません。")
        result = json.loads(row["result"]) if row["result"] else {}
        if row["status"] != "completed" or not result.get("sha256"):
            raise ValueError("完了した取り込みのジョブを指定してください（失敗したジョブには、確かめる OCR の本文がありません）。")
        return result["sha256"]

    def list(self, wiki_root: str) -> list[dict]:
        wiki = str(Path(wiki_root).expanduser().resolve())
        with self.connect() as db:
            ids = [row[0] for row in db.execute("SELECT id FROM ingest_jobs WHERE wiki=? ORDER BY created_at DESC", (wiki,))]
        return [self.status(wiki, identity) for identity in ids]

    def legacy_data(self) -> list[str]:
        """0.3.0 が残したデータの場所を返す。新しい版は開かず、消してよい（ADR-0013）。"""
        names = ("jobs.sqlite", "jobs.sqlite-wal", "jobs.sqlite-shm", "wikis")
        return [str(self.data_dir / name) for name in names if (self.data_dir / name).exists()]

    def unfinished(self) -> bool:
        with self.connect() as db:
            # 失敗したジョブは明示的な再試行でしか動かないので、設定の変更を塞がない（ADR-0013）。
            ingest = db.execute("SELECT 1 FROM ingest_jobs WHERE status IN ('queued','running','paused') LIMIT 1").fetchone()
            search = db.execute("SELECT 1 FROM search_requests WHERE status IN ('queued','running') AND updated_at>? LIMIT 1", (time.time() - 5,)).fetchone()
            return bool(ingest or search)

    def recover(self):
        # ワーカーの排他ロックを取得した実行者だけが呼ぶ。
        with self.connect() as db:
            db.execute("UPDATE ingest_jobs SET status='paused',phase='再開待ち',owner=NULL,updated_at=? WHERE status='running'", (time.time(),))
            db.execute("UPDATE search_requests SET status='queued' WHERE status='running'")

    def request_search(self, config: Config, wiki_root: str, query: str, limit: int) -> str:
        config.workspace(wiki_root)
        if not query.strip() or not 1 <= limit <= 50:
            raise ValueError("空でない質問と、1〜50 の limit を指定してください。")
        identity = uuid.uuid4().hex
        with self.connect() as db:
            db.execute("INSERT INTO search_requests(id,config,wiki,query,result_limit,updated_at) VALUES (?,?,?,?,?,?)",
                       (identity, json.dumps(config.snapshot(), ensure_ascii=False), wiki_root, query, limit, time.time()))
        return identity

    def search_status(self, identity: str) -> dict:
        with self.connect() as db:
            db.execute("UPDATE search_requests SET updated_at=? WHERE id=?", (time.time(), identity))
            row = db.execute("SELECT status,result,error FROM search_requests WHERE id=?", (identity,)).fetchone()
        if row is None:
            raise ValueError("検索要求が見つかりません。")
        return dict(row)

    def claim_search(self) -> dict | None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            # MCP が消失した検索要求を残し続けない。生存中の呼び出しは定期的に更新する。
            db.execute("DELETE FROM search_requests WHERE updated_at<?", (time.time() - 30,))
            row = db.execute("SELECT * FROM search_requests WHERE status='queued' AND updated_at>? LIMIT 1", (time.time() - 5,)).fetchone()
            if row is None:
                return None
            db.execute("UPDATE search_requests SET status='running' WHERE id=?", (row["id"],))
            return dict(row)

    def finish_search(self, identity: str, result=None, error=None):
        with self.connect() as db:
            db.execute("UPDATE search_requests SET status=?,result=?,error=? WHERE id=?",
                       ("failed" if error else "completed", json.dumps(result, ensure_ascii=False), error, identity))

    def drop_search(self, identity: str):
        with self.connect() as db:
            db.execute("DELETE FROM search_requests WHERE id=?", (identity,))

    def claim(self, owner: str) -> dict | None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM ingest_jobs WHERE status IN ('queued','paused') ORDER BY created_at LIMIT 1").fetchone()
            if row is None:
                return None
            db.execute("UPDATE ingest_jobs SET status='running',phase='開始',owner=?,attempts=attempts+1,error=NULL,updated_at=? WHERE id=?",
                       (owner, time.time(), row["id"]))
            return dict(row)

    def progress(self, identity: str, owner: str, phase: str, completed: int, total: int):
        with self.connect() as db:
            changed = db.execute("""UPDATE ingest_jobs SET phase=?,completed=?,total=?,updated_at=?
                WHERE id=? AND owner=? AND status='running'""",
                (phase, completed, total, time.time(), identity, owner)).rowcount
        if not changed:
            raise JobCancelled()

    def begin_publication(self, identity: str, owner: str):
        # 公開開始後はキャンセルを受理しない。台帳の切り替えとキャンセルの競合を防ぐ。
        with self.connect() as db:
            changed = db.execute("UPDATE ingest_jobs SET phase='索引の公開',publication_started=1,updated_at=? WHERE id=? AND owner=? AND status='running'",
                                 (time.time(), identity, owner)).rowcount
        if not changed:
            raise JobCancelled()

    def finish(self, identity: str, owner: str, *, result=None, error=None):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            # 作り直しの処理中に次の確認を受け付けていたら、終えずに待ちに戻す（ADR-0025 の「作り直しのジョブ」）
            if error is None and db.execute("UPDATE ingest_jobs SET status='queued',phase='確認の結果を当てる作り直しを再受付',"
                                            "rerun=0,result=?,owner=NULL,updated_at=? WHERE id=? AND owner=? AND status='running' "
                                            "AND kind='review' AND rerun=1",
                                            (json.dumps(result, ensure_ascii=False) if result is not None else None,
                                             time.time(), identity, owner)).rowcount:
                return
            db.execute("""UPDATE ingest_jobs SET status=?,phase=?,result=?,error=?,owner=NULL,updated_at=?
                WHERE id=? AND owner=? AND status='running'""",
                ("failed" if error else "completed", "失敗" if error else "完了",
                 json.dumps(result, ensure_ascii=False) if result is not None else None,
                 error, time.time(), identity, owner))

    def pause(self, owner: str):
        with self.connect() as db:
            db.execute("UPDATE ingest_jobs SET status='paused',owner=NULL,updated_at=? WHERE owner=? AND status='running'", (time.time(), owner))

    def cancel(self, wiki_root: str, job_id: str) -> dict:
        current = self.status(wiki_root, job_id)
        with self.connect() as db:
            changed = db.execute("""UPDATE ingest_jobs SET status='cancelled',phase='キャンセル',updated_at=?
                WHERE id=? AND status IN ('queued','paused','running','failed') AND publication_started=0""",
                (time.time(), job_id)).rowcount
        return {**self.status(current["wiki"], job_id), "cancellation_accepted": bool(changed)}

    def retry(self, wiki_root: str, job_id: str) -> dict:
        current = self.status(wiki_root, job_id)
        if current.get("kind") == "review":
            # 作り直しのジョブは、wiki と PDF ごとに 1 つだけ持つ。待っているか処理中のものがあれば、それを返す
            with self.connect() as db:
                row = db.execute("""SELECT id FROM ingest_jobs WHERE kind='review' AND wiki=? AND doc_id=? AND id!=?
                    AND status IN ('queued','running','paused') LIMIT 1""", (current["wiki"], current["doc_id"], job_id)).fetchone()
            if row:
                return self.status(current["wiki"], row[0])
        with self.connect() as db:
            db.execute("UPDATE ingest_jobs SET status='queued',phase='再試行待ち',owner=NULL,error=NULL,updated_at=? WHERE id=? AND status IN ('failed','cancelled')",
                       (time.time(), job_id))
        return self.status(current["wiki"], job_id)
