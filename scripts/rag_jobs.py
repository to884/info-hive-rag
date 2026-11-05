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
        self.path = self.data_dir / "jobs.sqlite"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS ingest_jobs (
                    id TEXT PRIMARY KEY, wiki TEXT NOT NULL, source TEXT NOT NULL,
                    source_page TEXT NOT NULL, title TEXT NOT NULL, size INTEGER, mtime_ns INTEGER,
                    config TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued',
                    phase TEXT NOT NULL DEFAULT '受付', completed INTEGER NOT NULL DEFAULT 0,
                    total INTEGER NOT NULL DEFAULT 0, attempts INTEGER NOT NULL DEFAULT 0,
                    owner TEXT, error TEXT, result TEXT, created_at REAL, updated_at REAL);
                CREATE TABLE IF NOT EXISTS search_requests (
                    id TEXT PRIMARY KEY, config TEXT NOT NULL, wiki TEXT NOT NULL,
                    query TEXT NOT NULL, result_limit INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued', result TEXT, error TEXT,
                    updated_at REAL NOT NULL);
            """)
            db.execute('BEGIN IMMEDIATE')
            columns={row['name'] for row in db.execute('PRAGMA table_info(ingest_jobs)')}
            if 'publication_started' not in columns:
                db.execute('ALTER TABLE ingest_jobs ADD COLUMN publication_started INTEGER NOT NULL DEFAULT 0')
                db.execute("UPDATE ingest_jobs SET publication_started=1 WHERE phase='索引の公開' OR status='completed'")

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
        root, relative, path = config.reference(source)
        if path.suffix.lower() != ".pdf" or not path.is_file():
            raise ValueError("読み取れる PDF ファイルを指定してください。")
        page = config.source_page(wiki, source_page)
        stat = path.stat()
        key = [os.path.normcase(str(wiki)), root, relative, stat.st_size, stat.st_mtime_ns,
               config.fingerprint, page, title.strip()]
        identity = hashlib.sha256(json.dumps(key, ensure_ascii=False).encode("utf-8")).hexdigest()
        now = time.time()
        with self.connect() as db:
            db.execute("""INSERT OR IGNORE INTO ingest_jobs
                (id,wiki,source,source_page,title,size,mtime_ns,config,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (identity, str(wiki), str(path), page, title.strip(), stat.st_size, stat.st_mtime_ns,
                 json.dumps(config.snapshot(), ensure_ascii=False), now, now))
        return self.status(str(wiki), identity)

    def status(self, wiki_root: str, job_id: str) -> dict:
        wiki = str(Path(wiki_root).expanduser().resolve())
        with self.connect() as db:
            row = db.execute("SELECT * FROM ingest_jobs WHERE id=? AND wiki=?", (job_id, wiki)).fetchone()
        if row is None:
            raise ValueError("この wiki に指定のジョブがありません。")
        result = dict(row)
        result.pop("config")
        result.pop("owner")
        result["job_id"] = result.pop("id")
        result["result"] = json.loads(result["result"]) if result["result"] else None
        return result

    def list(self, wiki_root: str) -> list[dict]:
        wiki = str(Path(wiki_root).expanduser().resolve())
        with self.connect() as db:
            ids = [row[0] for row in db.execute("SELECT id FROM ingest_jobs WHERE wiki=? ORDER BY created_at DESC", (wiki,))]
        return [self.status(wiki, identity) for identity in ids]

    def unfinished(self) -> bool:
        with self.connect() as db:
            ingest = db.execute("SELECT 1 FROM ingest_jobs WHERE status NOT IN ('completed','cancelled') LIMIT 1").fetchone()
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
        with self.connect() as db:
            db.execute("UPDATE ingest_jobs SET status='queued',phase='再試行待ち',owner=NULL,error=NULL,updated_at=? WHERE id=? AND status IN ('failed','cancelled')",
                       (time.time(), job_id))
        return self.status(current["wiki"], job_id)
