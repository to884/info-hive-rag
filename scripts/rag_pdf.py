"""PDF のページ検査、必要なページだけの OCR、再開可能な抽出。"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from contextlib import closing
from pathlib import Path

from rag_config import Config, write_json


def inspect_page(page) -> tuple[str, dict]:
    import pypdfium2 as pdfium

    with closing(page.get_textpage()) as textpage:
        text = textpage.get_text_range().replace("\r\n", "\n").replace("\r", "\n")
        visible = [c for c in text if not c.isspace()]
        bad = sum(c == "\ufffd" or 0xE000 <= ord(c) <= 0xF8FF for c in visible)
        cjk = sum("\u3040" <= c <= "\u30ff" or "\u3400" <= c <= "\u9fff" for c in visible)
        vertical = 0
        samples = min(textpage.count_chars(), 200)
        previous = None
        movements = []
        for i in range(samples):
            angle = pdfium.raw.FPDFText_GetCharAngle(textpage, i)
            if abs(math.sin(angle)) > 0.7:
                vertical += 1
            box = textpage.get_charbox(i)
            center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            if previous:
                dx, dy = abs(center[0] - previous[0]), abs(center[1] - previous[1])
                if dx + dy > 1:
                    movements.append(dy > dx * 2)
            previous = center
        is_vertical = samples > 5 and (vertical / samples > 0.4 or
                      (cjk > 5 and len(movements) > 5 and sum(movements) / len(movements) > 0.65))
    width, height = page.get_size()
    image_area = 0.0
    for image in page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_IMAGE]):
        left, bottom, right, top = image.get_bounds()
        image_area += max(0, right - left) * max(0, top - bottom)
    coverage = min(1.0, image_area / max(1, width * height))
    reason = "text"
    if bad / max(1, len(visible)) > 0.02:
        reason = "garbled"
    elif is_vertical:
        reason = "vertical"
    elif len(visible) < 20 and coverage > 0.1:
        reason = "scan"
    elif not visible:
        reason = "blank"
    return text, {"kind": reason, "chars": len(visible), "cjk_ratio": cjk / max(1, len(visible)),
                  "bad_ratio": bad / max(1, len(visible)), "image_coverage": coverage, "vertical": is_vertical}


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
