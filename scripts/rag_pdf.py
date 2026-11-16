"""PDF のページ検査、必要なページだけの OCR、再開可能な抽出。

文字の層の本文は pdfminer（縦書きの検出あり）で、ページの判定は PDFium で取る。どちらも wiki と同じ中身の
rag_pdf_layer が行う（ADR-0025）。OCR の結果は、文字の層の抽出の結果とは別の鍵で保存し、取り出し方を
変えても使い回す。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from contextlib import ExitStack, closing
from pathlib import Path

import rag_pdf_layer
from rag_config import (EXTRACT_REVISION, LEGACY_EXTRACT_REVISION, OCR_REVISION, OCR_STORE_REVISION, Config,
                        render_identity, text_layer_identity, write_json)
# ページの判定は、wiki と同じ中身の rag_pdf_layer に置く（ADR-0025）。rag_pdf.inspect_page の名前も残す
from rag_pdf_layer import inspect_page  # noqa: F401

# OCR に回す判定。ocr_mode が always なら、すべてのページを OCR する
OCR_KINDS = {"garbled", "scan", "vertical"}
# 移した OCR の結果の描画の識別。rag 0.5 以前の結果には、描画した pypdfium2 の版が記録されていない
UNKNOWN_RENDER = "unknown-rag-0.5"
# 移す元の、rag 0.5 の抽出の結果の置き場所の名前（<ocr_mode>-<ocr_dpi>-<OCR の版の 12 文字>-extract-v1）
LEGACY_CACHE = re.compile(r"^(auto|always)-(\d+)-([0-9a-f]{12})-" + re.escape(LEGACY_EXTRACT_REVISION) + "$")


def ocr_page(page, config: Config, work: Path, client=None) -> dict:
    """ページを描画して OCR し、NDLOCR-Lite の応答（text、line_count）を返す。"""
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
            response = client.recognize(image)
            return {"text": response["text"], "line_count": response.get("line_count")}
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
        return {"text": output.read_text(encoding="utf-8-sig").strip(), "line_count": None}


class OcrStore:
    """OCR の結果を、文字の層の抽出の結果とは別に保存する（ADR-0025 の「OCR の結果を別に保存する」）。

    鍵は、PDF の SHA-256（置き場所が文書ごと）、ページの番号、描画の識別（pypdfium2 の版と dpi）、
    NDLOCR-Lite の版である（Config.ocr_key）。ocr_mode と文字の層の識別は鍵に入れない。
    呼ぶ側は、文書の抽出のロック（Engine._extraction_lock）を持っている。
    """

    def __init__(self, config: Config, doc_id: str):
        self.config = config
        self.doc_root = config.extraction_root / doc_id
        self.folder = config.ocr_dir(doc_id)
        self.render = render_identity(config.ocr_dpi)
        self.marker = self.doc_root / "ocr" / "adopted.json"

    def path(self, number: int) -> Path:
        return self.folder / f"page-{number:06d}.json"

    def load(self, number: int) -> dict | None:
        """保存した結果を返す。今の描画の識別で使えない結果は None を返す。"""
        path = self.path(number)
        if not path.exists():
            return None
        entry = json.loads(path.read_text(encoding="utf-8"))
        if entry.get("store_revision") != OCR_STORE_REVISION or entry.get("ocr_revision") != OCR_REVISION:
            return None
        # 移した結果は、移したときの描画の識別と今が同じあいだだけ使う
        if entry.get("render") != self.render and entry.get("adopted_render") != self.render:
            return None
        return entry

    def save(self, number: int, response: dict) -> dict:
        entry = {"page": number, "text": response["text"], "line_count": response.get("line_count"),
                 "render": self.render, "ocr_revision": OCR_REVISION, "store_revision": OCR_STORE_REVISION,
                 # 文字ごとの確率と行の確信度は、段階 3 で足す。ないものは「確信度が不明」として扱う
                 "confidence": None, "saved_at": time.time()}
        write_json(self.path(number), entry)
        return entry

    def adopt_legacy(self) -> int:
        """rag 0.5 の抽出の結果（extract-v1）のうち OCR したページを、この保存先に移す。移したページの数を返す。

        移すのは、同じ PDF（置き場所が文書ごと）、同じ ocr_dpi、同じ OCR の版の結果だけである。ocr_mode は
        auto と always のどちらの結果も移す（同じ描画と OCR の版なら、OCR の結果は同じである）。描画した
        pypdfium2 の版は記録されていないので、描画の識別を UNKNOWN_RENDER とし、移したときの描画の識別を
        adopted_render に残す。0.4.0 が wiki ごとに置いた結果は移さない（0.5 の取り込みで、共有の置き場所に移してある）。

        印（adopted.json）は、ページを写す前に書く。元ごとに、移し始めたときの描画の識別と状態（copying、done）を
        記録し、写すページの adopted_render は印の値を使う。途中で止まったときは、描画の識別が印と同じときだけ
        続きを写す。描画の識別が変わったあとは、その元からは写さない（同じ元からもう一度は移さない）。
        """
        if not self.doc_root.is_dir():
            return 0
        record = json.loads(self.marker.read_text(encoding="utf-8")) if self.marker.exists() else {"sources": {}}
        moved = 0
        for source in sorted(self.doc_root.iterdir()):
            match = LEGACY_CACHE.match(source.name)
            if not source.is_dir() or not match:
                continue
            if int(match.group(2)) != self.config.ocr_dpi or match.group(3) != OCR_REVISION[:12]:
                continue
            known = record["sources"].get(source.name)
            if known is not None and (known["state"] == "done" or known["render"] != self.render):
                continue
            if known is None:
                known = record["sources"][source.name] = {"render": self.render, "state": "copying",
                                                          "started_at": time.time()}
                write_json(self.marker, record)  # 写す前に、移し始めたことと描画の識別を残す
            pages = []
            for saved in sorted(source.glob("page-*.json")):
                entry = json.loads(saved.read_text(encoding="utf-8"))
                number = int(entry["page"])
                if not entry.get("ocr") or self.path(number).exists():
                    continue
                write_json(self.path(number), {
                    "page": number, "text": entry["text"], "line_count": None, "render": UNKNOWN_RENDER,
                    "adopted_render": known["render"], "adopted_from": source.name, "ocr_revision": OCR_REVISION,
                    "store_revision": OCR_STORE_REVISION, "confidence": None, "saved_at": time.time()})
                pages.append(number)
            known.update({"state": "done", "pages": rag_pdf_layer.page_ranges(pages), "done_at": time.time()})
            write_json(self.marker, record)
            moved += len(pages)
        return moved


def _judgement_path(cache: Path, number: int) -> Path:
    return cache / f"judge-{number:06d}.json"


def extract_pages(data: bytes, config: Config, cache: Path, progress=None, ocr_client=None) -> list[dict]:
    """ページごとに判定し、OCR が要るページは OCR して、ページの最終の本文を返す。途中から再開できる。

    判定（judge-*.json）は OCR の前に保存し、ページの結果（page-*.json）は本文が決まってから保存する。
    途中で止まっても、page_report は判定まで済んだページを返せる。ocr_client を渡すと、それで OCR する
    （試験で OCR の代わりのスタブを使う）。
    """
    import pypdfium2 as pdfium

    # 宣言した版と、実際に読み込んだ版が同じかを照らす（違えば止まる）
    rag_pdf_layer.extraction_identity()
    cache.mkdir(parents=True, exist_ok=True)
    doc_id = hashlib.sha256(data).hexdigest()
    count = rag_pdf_layer.pdfium_page_count(data)
    key = config.extraction_key
    saved = {}
    for path in cache.glob("page-*.json"):
        entry = json.loads(path.read_text(encoding="utf-8"))
        if entry.get("extraction_key") == key:  # 鍵の違う記録（置き場所の取り違えなど）は使わない
            saved[int(entry["page"])] = entry
    missing = [number for number in range(1, count + 1) if number not in saved]
    if not missing:
        return [saved[number] for number in range(1, count + 1)]
    store = OcrStore(config, doc_id)
    store.adopt_legacy()
    with ExitStack() as stack:
        document = stack.enter_context(pdfium.PdfDocument(data))
        ocr = ocr_client
        for entry in rag_pdf_layer.analyze_pages(data, start=missing[0]):
            number = entry["page"]
            if progress:
                progress("PDF の抽出", number - 1, count)
            if number in saved:
                continue
            kind = entry["kind"]
            needs_ocr = kind in OCR_KINDS or config.ocr_mode == "always"
            write_json(_judgement_path(cache, number), {
                "page": number, "extraction_key": key, "kind": kind, "quality": entry["quality"], "garble": entry["garble"],
                "needs_check": entry["needs_check"], "text_sha256": entry["text_sha256"],
                "text_empty": not entry["text"], "needs_ocr": needs_ocr})
            ocr_source = None
            if needs_ocr:
                if config.ocr_mode == "off":
                    raise ValueError(f"{number} ページ目に OCR が必要です。ocr_mode を auto にしてください。")
                found = store.load(number)
                ocr_source = "adopted" if found and found.get("render") == UNKNOWN_RENDER else "stored"
                if found is None:
                    if ocr is None:
                        from rag_ocr_client import OcrClient
                        ocr = stack.enter_context(OcrClient(config))
                    with closing(document[number - 1]) as page:
                        found = store.save(number, ocr_page(page, config, cache, client=ocr))
                    ocr_source = "ocr"
                text = found["text"]
            else:
                # OCR しないページの本文は、wiki の wiki_pdf.py と同じ決まりにする（blank は本文を出さない）
                text = entry["text"] if kind in ("text", "vertical") else ""
            page_entry = {"page": number, "extraction_key": key, "text": text, "quality": entry["quality"], "kind": kind,
                          "ocr": needs_ocr, "ocr_empty": needs_ocr and not text.strip(), "ocr_source": ocr_source,
                          "garble": entry["garble"], "needs_check": entry["needs_check"],
                          "text_sha256": entry["text_sha256"]}
            write_json(cache / f"page-{number:06d}.json", page_entry)
            saved[number] = page_entry
            if progress:
                progress("PDF の抽出", number, count)
    return [saved[number] for number in range(1, count + 1)]


def page_report(data: bytes, config: Config, cache: Path) -> dict:
    """保存したページごとの記録から、ページの情報を作る（ADR-0025 の「ページの情報を返す」）。

    成功、失敗、取り込み済み（unchanged）のどの場合も、同じこの関数で作る。一覧は、続いた番号を
    まとめた範囲の、すべての範囲を持つ（返すときに、Jobs.status が 200 個ずつに切る）。
    """
    count = rag_pdf_layer.pdfium_page_count(data)
    key = config.extraction_key
    judged, done = {}, {}
    for path in cache.glob("judge-*.json"):
        entry = json.loads(path.read_text(encoding="utf-8"))
        if entry.get("extraction_key") == key:
            judged[int(entry["page"])] = entry
    for path in cache.glob("page-*.json"):
        entry = json.loads(path.read_text(encoding="utf-8"))
        if entry.get("extraction_key") == key:
            done[int(entry["page"])] = entry

    def listed(numbers) -> dict:
        numbers = sorted(set(numbers))
        return {"count": len(numbers), "ranges": rag_pdf_layer.page_ranges(numbers)}

    lists = {f"kind:{kind}": listed(n for n, e in judged.items() if e["kind"] == kind) for kind in rag_pdf_layer.KINDS}
    lists["needs_check"] = listed(n for n, e in judged.items() if e["needs_check"])
    lists["empty_text"] = listed(n for n, e in judged.items() if e["text_empty"])
    lists["ocr"] = listed(n for n, e in done.items() if e.get("ocr"))
    lists["ocr_empty"] = listed(n for n, e in done.items() if e.get("ocr_empty"))
    # 判定は済んだが、本文が決まっていないページ（OCR の途中で止まった、ocr_mode が off で止まったなど）
    lists["pending"] = listed(n for n in judged if n not in done)
    return {"pages": count, "judged": len(judged), "processed": len(done), "complete": len(done) == count,
            "lists": lists, "extraction": text_layer_identity(), "extract_revision": EXTRACT_REVISION,
            "ocr": {"mode": config.ocr_mode, "dpi": config.ocr_dpi, "revision": OCR_REVISION,
                    "render": render_identity(config.ocr_dpi), "store": config.ocr_key},
            "confidence": None}


def render_page_report(report: dict | None, start: int = 0, limit: int = rag_pdf_layer.LIST_LIMIT) -> dict | None:
    """保存した page_report の一覧を、ページの一覧の形（count、ranges、next）にして返す。"""
    if not report:
        return report
    if start < 0:
        raise ValueError("page_report_from は 0 以上にしてください。")
    lists = {}
    for name, item in report["lists"].items():
        ranges = item["ranges"]
        chosen = ranges[start:start + limit]
        lists[name] = {"count": item["count"], "ranges": ",".join(chosen),
                       "next": start + limit if start + limit < len(ranges) else None}
    return {**report, "lists": lists, "list_from": start}


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
