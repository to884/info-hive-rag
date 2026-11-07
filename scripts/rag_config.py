"""PC ごとの設定と、原資料から独立した派生データの配置。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import sys
import time
import uuid
from pathlib import Path


MODELS = {
    "cl-nagoya/ruri-v3-30m": 256,
    "cl-nagoya/ruri-v3-70m": 384,
    "cl-nagoya/ruri-v3-130m": 512,
    "cl-nagoya/ruri-v3-310m": 768,
}
DEFAULT_MODEL = "cl-nagoya/ruri-v3-130m"
DEFAULT_REVISION = "e3114c6ee10dbab8b4b235fbc6dcf9dd4d5ac1a6"
OCR_REVISION = "e57f7c6bd9cfb3de5f12f008e71ad2122bd44bfd"
MODEL_REVISIONS = {
    "cl-nagoya/ruri-v3-30m": "24899e5de370b56d179604a007c0d727bf144504",
    "cl-nagoya/ruri-v3-70m": "07a8b0aba47d29d2ca21f89b915c1efe2c23d1cc",
    DEFAULT_MODEL: DEFAULT_REVISION,
}


def app_data() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library/Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    return base / "info-hive/rag"


def config_path() -> Path:
    return Path(os.environ.get("INFO_HIVE_RAG_CONFIG", app_data() / "config.json")).expanduser().resolve()


def _network_drive(path: Path) -> bool:
    """Windows で、パスのドライブ文字が、利用者が割り当てたネットワークドライブなら True を返す。"""
    if not path.drive or not path.drive.endswith(":"):
        return False
    import ctypes
    return ctypes.windll.kernel32.GetDriveTypeW(path.drive + "\\") == 4  # DRIVE_REMOTE


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Windows で短時間だけ共有違反になる場合に、保存途中のデータを公開せず再試行する。
    for attempt in range(8):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(.01 * 2 ** attempt, .2))


class Config:
    def __init__(self, values: dict | None = None):
        self.path = config_path()
        if values is None:
            values = json.loads(self.path.read_text(encoding="utf-8-sig")) if self.path.exists() else {}
        self.values = values
        self.data_dir = Path(values.get("data_dir", app_data() / "data")).expanduser().resolve()
        self.model = values.get("embedding_model", DEFAULT_MODEL)
        if self.model not in MODELS:
            raise ValueError("embedding_model は Ruri v3 の 30m、70m、130m、310m から選んでください。")
        self.revision = values.get("embedding_revision", MODEL_REVISIONS.get(self.model, "main"))
        self.model_cache_dir = Path(values.get("model_cache_dir", self.data_dir / "models")).expanduser().resolve()
        self.uv_cache_dir = Path(values.get("uv_cache_dir", self.data_dir / "uv-cache")).expanduser().resolve()
        self.dimension = MODELS[self.model]
        self.chunk_chars = int(values.get("chunk_chars", 800))
        self.overlap = int(values.get("chunk_overlap", 100))
        self.batch_size = int(values.get("batch_size", 16))
        self.threads = int(values.get("cpu_threads", max(1, min(os.cpu_count() or 1, 8))))
        self.ocr_mode = values.get("ocr_mode", "auto")
        self.ocr_dpi = int(values.get("ocr_dpi", 150))
        self.reranker = values.get("reranker", "off")
        self.rerank_limit = int(values.get("rerank_limit", 8))
        threshold = values.get("min_rerank_score")
        self.min_rerank_score = float(threshold) if threshold is not None else None
        self.search_nprobes = int(values.get("search_nprobes", 20))
        self.search_refine_factor = int(values.get("search_refine_factor", 4))
        self.ann_min_rows = int(values.get("ann_min_rows",256))
        self.ann_rebuild_growth = float(values.get("ann_rebuild_growth",4))
        self.index_cleanup_age_days = int(values.get("index_cleanup_age_days",7))
        if not 100 <= self.chunk_chars <= 1000 or not 0 <= self.overlap < self.chunk_chars // 2:
            raise ValueError("chunk_chars は 100〜1000、chunk_overlap はその半分未満にしてください。")
        if not 1 <= self.batch_size <= 128 or not 1 <= self.threads <= 256:
            raise ValueError("batch_size は 1〜128、cpu_threads は 1〜256 にしてください。")
        if self.ocr_mode not in {"auto", "always", "off"} or not 72 <= self.ocr_dpi <= 300:
            raise ValueError("ocr_mode は auto / always / off、ocr_dpi は 72〜300 にしてください。")
        if self.reranker not in {"off", "xsmall-int8", "tiny-int8"} or not 1 <= self.rerank_limit <= 50:
            raise ValueError("reranker は off / xsmall-int8 / tiny-int8、rerank_limit は 1〜50 にしてください。")
        if self.min_rerank_score is not None and (not math.isfinite(self.min_rerank_score) or self.reranker == "off"):
            raise ValueError("min_rerank_score は有限の数値とし、リランカーを有効にしてください。")
        if not 1 <= self.search_nprobes <= 4096 or not 1 <= self.search_refine_factor <= 100:
            raise ValueError("search_nprobes は 1〜4096、search_refine_factor は 1〜100 にしてください。")
        if not 1<=self.ann_min_rows<=1000000 or not math.isfinite(self.ann_rebuild_growth) or not 2<=self.ann_rebuild_growth<=16 or not 1<=self.index_cleanup_age_days<=36500:
            raise ValueError('ann_min_rows は1〜1000000、ann_rebuild_growth は2〜16、index_cleanup_age_days は1〜36500にしてください。')

    def snapshot(self) -> dict:
        values = {key: value for key, value in self.values.items() if key != "reference_roots"}
        return {**values, "data_dir": str(self.data_dir),
                "embedding_model": self.model, "embedding_revision": self.revision,
                "model_cache_dir": str(self.model_cache_dir), "uv_cache_dir": str(self.uv_cache_dir)}

    def source_page(self, wiki: Path, source_page: str) -> str:
        page = (wiki / source_page).resolve()
        if not page.is_relative_to(wiki / "sources") or page.suffix.lower() != ".md":
            raise ValueError("source_page は wiki 内の sources/<スラッグ>.md にしてください。")
        return page.relative_to(wiki).as_posix()

    @property
    def fingerprint(self) -> str:
        policy = ["rag-v2", self.model, self.revision, self.dimension, self.chunk_chars,
                  self.overlap, self.ocr_mode, self.ocr_dpi, OCR_REVISION, "sudachi-C-normalized", "ruri-prefix-v1", "native-context-token-budget-v1"]
        return hashlib.sha256(json.dumps(policy).encode("utf-8")).hexdigest()

    def workspace(self, wiki_root: str) -> Path:
        wiki = Path(wiki_root).expanduser().resolve()
        if not (wiki / "SCHEMA.md").is_file():
            raise ValueError("wiki_root に SCHEMA.md がありません。先に wiki を初期化してください。")
        identity = hashlib.sha256(os.path.normcase(str(wiki)).encode("utf-8")).hexdigest()[:24]
        return self.data_dir / "wikis-v2" / identity

    def pdf_path(self, source: str) -> Path:
        """登録する PDF の絶対パスを確かめ、解決したパスを返す。中身は読まない（ADR-0013）。"""
        text = str(source).strip()
        if not text:
            raise ValueError("PDF の絶対パスを指定してください。")
        # UNC とデバイスのパスは、解決する前に拒む。知らないサーバーに接続させないため。
        if text.startswith("\\\\") or (sys.platform == "win32" and text.startswith("//")):
            raise ValueError("UNC パスとデバイスのパスは指定できません。ネットワークの PDF は、ドライブ文字を割り当ててから指定してください。")
        path = Path(text).expanduser()
        if not path.is_absolute():
            raise ValueError("PDF は絶対パスで指定してください。")
        try:
            resolved = path.resolve(strict=True)
            info = os.lstat(resolved)
        except OSError:
            raise ValueError("PDF が見つかりません。絶対パスを確かめてください。") from None
        if not stat.S_ISREG(info.st_mode) or resolved.suffix.lower() != ".pdf":
            raise ValueError("読み取れる PDF ファイル（拡張子 .pdf）を指定してください。")
        if sys.platform == "win32" and str(resolved).startswith("\\\\") and not _network_drive(path):
            # ローカルのフォルダーに置かれたリンクが、ネットワーク上のファイルを指している。
            raise ValueError("ローカルのフォルダーから、ネットワーク上のファイルへのリンクはたどりません。")
        for folder in (self.data_dir, self.model_cache_dir, self.uv_cache_dir):
            if resolved == folder or resolved.is_relative_to(folder):
                raise ValueError("rag の派生データやキャッシュの中の PDF は登録できません。")
        return resolved
