# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "mcp>=1.28,<2", "lancedb>=0.25,<0.30", "pypdfium2>=5,<6",
#   "sentence-transformers>=5,<6", "sudachipy>=0.6,<0.8", "sudachidict-core",
#   "pyyaml>=6,<7", "filelock>=3,<5", "psutil>=7,<8", "torch==2.8.0", "numpy>=2,<3", "pillow>=11,<14", "onnxruntime>=1.31,<2",
# ]
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""wiki の PDF インジェストと検索に使う、ローカル stdio MCP サーバー。"""

from __future__ import annotations

import json
import asyncio
import sys
import threading
import time
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

from rag_config import Config, config_path, write_json
from rag_index import Engine
from rag_jobs import Jobs
from rag_worker import WorkerHost

for stream in (sys.stdin, sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

_engine = None
_config = None
_jobs = None
_host = None
_lock = threading.RLock()


def configuration() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config


def jobs() -> Jobs:
    global _jobs
    if _jobs is None:
        _jobs = Jobs(configuration().data_dir)
    return _jobs


@asynccontextmanager
async def lifespan(_server):
    global _host
    _host = WorkerHost(jobs())
    _host.start()
    try:
        yield {}
    finally:
        if _host:
            _host.close()


server = FastMCP("info-hive-rag", lifespan=lifespan)


def engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = Engine(configuration())
    return _engine


@server.tool()
def configure(data_dir: str = "", embedding_model: str = "", retrieval: dict | None = None,
              ocr_mode: str = "", ocr_dpi: int = 0) -> dict:
    """派生データの保存先、埋め込みのモデル、検索の設定、OCR の設定（ocr_mode は auto / always / off、ocr_dpi は 72〜300）を、
    この PC の設定に保存する。指定しない項目は今の設定のまま。PDF は登録のときに絶対パスで渡す。
    reingest_required が true なら取り込み済みの PDF の入れ直しが要り、extraction_changed が true なら入れ直しで OCR もやり直す。"""
    global _engine, _jobs, _host, _config
    with _lock:
        if jobs().unfinished():
            raise ValueError("未完了ジョブがあります。完了させるかキャンセルしてから設定を変更してください。")
        current = configuration()
        embedding_model = embedding_model or current.model
        # 0.3.0 の reference_roots は使わないので、保存し直すときに消す（ADR-0013）。
        values = {key: value for key, value in current.values.items() if key != "reference_roots"}
        values["embedding_model"] = embedding_model
        if embedding_model != current.model:
            values.pop("embedding_revision", None)
        if data_dir:
            values["data_dir"] = data_dir
        if retrieval is not None:
            allowed = {"reranker", "rerank_limit", "min_rerank_score", "search_nprobes", "search_refine_factor"}
            if set(retrieval) - allowed:
                raise ValueError("retrieval に未対応の設定名があります。")
            values.update(retrieval)
        if ocr_mode:
            values["ocr_mode"] = ocr_mode
        if ocr_dpi:
            values["ocr_dpi"] = ocr_dpi
        checked = Config(values)
        if _host:
            _host.close()
        write_json(config_path(), values)
        _config = checked
        _engine, _jobs = None, Jobs(checked.data_dir)
        _host = WorkerHost(_jobs)
        _host.start()
        # 指紋が変われば取り込み済みの PDF の入れ直しが要る。抽出の設定も変われば OCR もやり直す（ADR-0014）。
        return {"config": str(config_path()), "data_dir": str(checked.data_dir), "embedding_model": checked.model,
                "ocr_mode": checked.ocr_mode, "ocr_dpi": checked.ocr_dpi,
                # 保存先を変えると、前の保存先の索引も抽出の結果も引き継がない。
                "reingest_required": checked.fingerprint != current.fingerprint or checked.data_dir != current.data_dir,
                "extraction_changed": checked.extraction_key != current.extraction_key or checked.data_dir != current.data_dir}


@server.tool()
def register_pdf(wiki_root: str, source: str, source_page: str, title: str = "") -> dict:
    """wiki の PDF 取り込みを受け付け、永続ジョブを返す。source は利用者が指定した PDF の絶対パス。queued は索引化完了ではない。原本はコピーしない。"""
    with _lock:
        return jobs().register(configuration(), wiki_root, source, source_page, title)


@server.tool()
def ingest_pdf(wiki_root: str, source: str, source_page: str, title: str = "") -> dict:
    """register_pdf の互換名。ジョブを受け付けて返し、完了は job_status で確認する。"""
    return register_pdf(wiki_root, source, source_page, title)


@server.tool()
async def job_status(wiki_root: str, job_id: str, wait_seconds: int = 0) -> dict:
    """進捗と完了時の抽出 Markdown を返す。wait_seconds=1〜30 で完了を短時間待てる。"""
    if not 0 <= wait_seconds <= 30:
        raise ValueError("wait_seconds は 0〜30 にしてください。")
    deadline = time.monotonic() + wait_seconds
    while True:
        result = jobs().status(wiki_root, job_id)
        if result["status"] in {"completed", "failed", "cancelled"} or time.monotonic() >= deadline:
            return result
        await asyncio.sleep(.1)


@server.tool()
def cancel_job(wiki_root: str, job_id: str) -> dict:
    """明示キャンセルし、次回接続時の自動再開から外す。公開開始後はキャンセルを受理しない。"""
    return jobs().cancel(wiki_root, job_id)


@server.tool()
def retry_job(wiki_root: str, job_id: str) -> dict:
    """失敗またはキャンセルしたジョブを明示的に再試行し、チェックポイントを再利用する。"""
    return jobs().retry(wiki_root, job_id)


@server.tool()
async def search(wiki_root: str, query: str, limit: int = 10) -> dict:
    """取り込み済み PDF を BM25 とベクトルで検索し、RRF 順位、本文、ページ、出典を返す。原本を読み直さない。"""
    identity = jobs().request_search(configuration(), wiki_root, query, limit)
    try:
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            result = jobs().search_status(identity)
            if result["status"] == "completed":
                return json.loads(result["result"])
            if result["status"] == "failed":
                raise ValueError(result["error"])
            await asyncio.sleep(.02)
        raise TimeoutError("検索の待機が時間切れになりました。索引化中の場合は完了後に再検索してください。")
    finally:
        jobs().drop_search(identity)


@server.tool()
def status(wiki_root: str) -> dict:
    """取り込み済みの件数と、中断・失敗した処理を確認する。"""
    result = engine().status(wiki_root)
    result["index_jobs"] = result.pop("jobs")
    result["jobs"] = jobs().list(wiki_root)
    legacy = jobs().legacy_data()
    if legacy:
        result["legacy_data"] = {"paths": legacy,
                                 "message": "rag 0.3.0 のデータが残っています。新しい版は読まないので、消してかまいません。"}
    return result


if __name__ == "__main__":
    server.run(transport="stdio")
