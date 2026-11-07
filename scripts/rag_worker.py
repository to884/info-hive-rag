"""MCP の接続中だけ動くワーカーと、その子プロセスの管理。"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import psutil
from filelock import FileLock, Timeout

from rag_config import Config
from rag_jobs import JobCancelled, Jobs


def stop_tree(pid: int):
    """自分が起動したプロセスと、その子孫だけを停止する。"""
    try:
        process = psutil.Process(pid)
        descendants = process.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    for child in reversed(descendants):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        process.kill()
    except psutil.NoSuchProcess:
        pass
    psutil.wait_procs(descendants + [process], timeout=5)


class WorkerHost:
    def __init__(self, jobs: Jobs):
        self.jobs = jobs
        self.stopped = threading.Event()
        self.process = None
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._supervise, name="rag-worker-host", daemon=True)
        self.thread.start()

    def _supervise(self):
        log_path = self.jobs.data_dir / "worker.log"
        while not self.stopped.is_set():
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
            try:
                with log_path.open("a", encoding="utf-8") as log:
                    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--data-dir", str(self.jobs.data_dir)],
                                             stdin=subprocess.PIPE, stdout=log, stderr=log, env=env, creationflags=flags)
                    self.process = child
                    while not self.stopped.wait(.1) and child.poll() is None:
                        pass
                    if self.stopped.is_set():
                        child.stdin.close()
                        try:
                            child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            stop_tree(child.pid)
                            child.wait(timeout=5)
                    else:
                        child.stdin.close()
                        if child.returncode not in (None,0):
                            log.write(f"ワーカーが終了しました（終了コード {child.returncode}）。未完了の処理は再開します。\n")
                            log.flush()
            except Exception as error:
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"ワーカーを起動できませんでした: {error}\n")
            if not self.stopped.is_set():
                self.stopped.wait(1)

    def close(self):
        self.stopped.set()
        if self.thread:
            self.thread.join(timeout=12)


def run(data_dir: str):
    jobs = Jobs(Path(data_dir))
    owner = uuid.uuid4().hex
    # 同じ保存先では、複数の MCP 接続があってもモデルを持つワーカーを一つにする。
    try:
        lock = FileLock(str(jobs.data_dir / "worker.lock"), timeout=0)
        lock.acquire()
    except Timeout:
        return

    if os.name == "nt":
        import ctypes
        import msvcrt
        from ctypes import wintypes
        input_handle = wintypes.HANDLE(msvcrt.get_osfhandle(sys.stdin.fileno()))
        peek = ctypes.WinDLL("kernel32", use_last_error=True).PeekNamedPipe
        peek.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        peek.restype = wintypes.BOOL

    def parent_closed():
        # 親が強制終了してもパイプの EOF を受け取れる。OCR の uv とその子孫も停止する。
        if os.name == "nt":
            # ブロックする CRT の stdin 読み取りと、NumPy 等の DLL 初期化を競合させない。
            while peek(input_handle, None, 0, None, None, None):
                time.sleep(.1)
        else:
            sys.stdin.buffer.read()
        try:
            jobs.pause(owner)
        finally:
            for child in reversed(psutil.Process().children(recursive=True)):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            os._exit(0)

    threading.Thread(target=parent_closed, name="rag-parent-watch", daemon=True).start()
    try:
        jobs.recover()
        engine, engine_key = None, None
        def get_engine(values):
            nonlocal engine, engine_key
            from rag_index import Engine
            config = Config(json.loads(values))
            key = json.dumps(config.snapshot(), sort_keys=True)
            if key != engine_key:
                engine, engine_key = Engine(config), key
            return engine
        while True:
            request = jobs.claim_search()
            if request:
                try:
                    result = get_engine(request["config"]).search(request["wiki"], request["query"], request["result_limit"])
                    jobs.finish_search(request["id"], result=result)
                except Exception as error:
                    jobs.finish_search(request["id"], error=str(error))
                continue
            job = jobs.claim(owner)
            if job is None:
                time.sleep(.02)
                continue
            try:
                source = Path(job["source"])
                stat = source.stat()
                if (stat.st_size, stat.st_mtime_ns) != (job["size"], job["mtime_ns"]):
                    raise ValueError("登録後に PDF が変わりました。新しいジョブで取り込んでください。")
                result = get_engine(job["config"]).ingest_pdf(job["wiki"], job["source"], job["source_page"], job["title"],
                    progress=lambda phase, completed, total: jobs.progress(job["id"], owner, phase, completed, total),
                    before_publish=lambda: jobs.begin_publication(job["id"], owner),
                    expected=(job["dev"], job["ino"], job["size"], job["mtime_ns"]))
                jobs.finish(job["id"], owner, result=result)
            except JobCancelled:
                pass
            except Exception as error:
                jobs.finish(job["id"], owner, error=str(error))
    finally:
        jobs.pause(owner)
        lock.release()


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="接続中の RAG ジョブを処理する。")
    parser.add_argument("--data-dir", required=True)
    run(parser.parse_args().data_dir)
