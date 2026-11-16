"""PDF のページ検査、必要なページだけの OCR、再開可能な抽出。"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from contextlib import closing
from pathlib import Path

from rag_config import Config, write_json
# ページの判定は、wiki と同じ中身の rag_pdf_layer に置く（ADR-0025）。rag_index や試験は rag_pdf.inspect_page として使う
from rag_pdf_layer import inspect_page  # noqa: F401


def ocr_page(page, config: Config, work: Path, client=None) -> str:
    # 原本ではなく、メモリ上の PDF から描画した一時画像を渡す。
    with tempfile.TemporaryDirectory(prefix="ocr-", dir=work, ignore_cleanup_errors=True) as directory:
        folder = Path(directory)
        image = folder / "page.png"
        bitmap = page.render(scale=config.ocr_dpi / 72)
        try:
            bitmap.to_pil().save(image)
        finally:
            bitmap.close()
        if client is not None:
            return client.recognize(image)['text']
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
               "UV_CACHE_DIR": str(config.uv_cache_dir)}
        runner = Path(__file__).with_name("rag_ocr.py")
        result = subprocess.run(["uv", "run", "--no-project", "--script", str(runner),
                                 "--sourceimg", str(image), "--output", str(folder), "--device", "cpu"],
                                stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace",
                                capture_output=True, env=env, timeout=1800)
        if result.returncode:
            raise RuntimeError("NDLOCR-Lite に失敗しました。再投入すると抽出済みのページから再開します。\n" + result.stderr[-2000:])
        output = folder / "page.txt"
        if not output.is_file():
            raise RuntimeError("NDLOCR-Lite のテキスト出力がありません。")
        text = output.read_text(encoding="utf-8-sig").strip()
        return text


def extract_pages(data: bytes, config: Config, cache: Path, progress=None) -> list[dict]:
    import pypdfium2 as pdfium

    cache.mkdir(parents=True, exist_ok=True)
    pages = []
    from rag_ocr_client import OcrClient
    with pdfium.PdfDocument(data) as document, OcrClient(config) as ocr:
        for number in range(len(document)):
            saved = cache / f"page-{number + 1:06d}.json"
            if progress:
                progress("PDF の抽出", number, len(document))
            if saved.exists():
                pages.append(json.loads(saved.read_text(encoding="utf-8")))
                continue
            with closing(document[number]) as page:
                text, quality = inspect_page(page)
                needs_ocr = quality["kind"] in {"garbled", "scan", "vertical"} or config.ocr_mode == "always"
                if needs_ocr:
                    if config.ocr_mode == "off":
                        raise ValueError(f"{number + 1} ページ目に OCR が必要です。ocr_mode を auto にしてください。")
                    text = ocr_page(page, config, cache,client=ocr)
                entry = {"page": number + 1, "text": text, "quality": quality, "ocr": needs_ocr,
                         "ocr_empty":needs_ocr and not text.strip()}
                write_json(saved, entry)
                pages.append(entry)
                if progress:
                    progress("PDF の抽出", number + 1, len(document))
    return pages


def chunks_for(pages: list[dict], title: str, config: Config) -> list[dict]:
    import re

    chunks = []
    section = ""
    for page in pages:
        # ページ境界を越えず、見出し・段落・句点の順で分割する。
        blocks = re.split(r"\n\s*\n|(?=^#{1,6}\s)", page["text"], flags=re.M)
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            if re.match(r"^#{1,6}\s", block):
                section = block.splitlines()[0].lstrip("# ")
            start = 0
            while start < len(block):
                end = min(len(block), start + config.chunk_chars)
                if end < len(block):
                    boundary = max(block.rfind(c, start + config.chunk_chars // 2, end) for c in "。！？\n")
                    if boundary >= 0:
                        end = boundary + 1
                text = block[start:end].strip()
                if text:
                    chunks.append({"page": page["page"], "section": section, "text": text,
                                   "embedding_text": f"{title}\n{section}\n{text}"})
                if end == len(block):
                    break
                start = max(start + 1, end - config.overlap)
    return chunks
