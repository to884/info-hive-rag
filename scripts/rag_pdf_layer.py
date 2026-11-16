"""PDF の文字の層の取り出しと、ページの判定（ADR-0025）。

このファイルは、wiki（info-hive-wiki の skills/info-hive/scripts/wiki_pdf_layer.py）と rag
（info-hive-rag の scripts/rag_pdf_layer.py）に、同じ中身で置く。片方を直したら、もう片方も同じに直す。
info-hive-dev の tests/contract/test_pdf_layer.py が、2 つのファイルが同じバイト列であることを確かめる。
wiki は rag のスクリプトを読み込まず、rag も wiki のスクリプトを読み込まないので、同じ中身を 2 か所に持つ。

- 文字の層の本文: pdfminer.six を LAParams(detect_vertical=True) で使い、文書を 1 回だけ読んで、
  ページごとに本文を受け取る。
- ページの判定: PDFium（pypdfium2）の文字の層から数値を数える（rag の inspect_page と同じ）。
  判定のためだけに使い、PDFium の本文は文字の層の本文にしない。
- 本文を毎回同じにする: pdfminer の group_textboxes は、距離の同じテキストボックスの組の順を id(obj)
  （メモリーの番地）で決めるので、プロセスによって本文の行の並びが変わる。ここでは、同じ手順で、組の順を
  ページの中で決まった番号で決める。モジュールの変数を書きかえないので、スレッドから同時に呼んでよい。
- 依存は関数の中で読み込む。pdfminer.six と pypdfium2 の版は、呼ぶ側のスクリプトの宣言で == に固定する。
"""

from __future__ import annotations

import hashlib
import heapq
import io
import math
import re
from collections.abc import Iterator
from contextlib import closing

# 取り出し方か判定を変えるときは、この値を上げる（wiki と rag で同じ値にする）
EXTRACT_REVISION = "extract-v2"
# 距離の同じ組の順の決め方の版
ORDER_REVISION = "stable-order-v1"
LAPARAMS = {"detect_vertical": True}
PDFMINER_VERSION = "20260107"
PYPDFIUM2_VERSION = "5.15.2"
# 確かめるべきページの条件（ADR-0025 の「確かめるべきページ」）
CHECK_IMAGE_COVERAGE = 0.5
# ページの一覧で、1 回に返す範囲の数
LIST_LIMIT = 200
KINDS = ("text", "vertical", "garbled", "scan", "blank")
CID = re.compile(r"\(cid:\d+\)")


def extraction_identity() -> dict:
    """抽出の識別。保存した結果と索引の鍵に入れ、wiki のログと rag の job_status に書く。

    宣言した版と、実際に読み込んだ版が違えば止まる（識別と中身がずれないようにする）。
    """
    from importlib.metadata import version

    found = {"pdfminer.six": version("pdfminer.six"), "pypdfium2": version("pypdfium2")}
    expected = {"pdfminer.six": PDFMINER_VERSION, "pypdfium2": PYPDFIUM2_VERSION}
    if found != expected:
        raise RuntimeError(f"pdfminer.six と pypdfium2 の版が、抽出の識別と違う（期待 {expected}、実際 {found}）")
    return {"extract_revision": EXTRACT_REVISION, "order_revision": ORDER_REVISION,
            "pdfminer.six": PDFMINER_VERSION, "laparams": dict(LAPARAMS), "pypdfium2": PYPDFIUM2_VERSION}


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def inspect_page(page) -> tuple[str, dict]:
    """PDFium のページを判定する。返す本文は PDFium の文字の層で、判定の材料と rag の今の本文に使う。"""
    import pypdfium2 as pdfium

    with closing(page.get_textpage()) as textpage:
        text = normalize_newlines(textpage.get_text_range())
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


def garble_counts(text: str) -> dict:
    """pdfminer の本文で、復号できなかった文字を数える。私用領域の文字は数えない。"""
    return {"cid": len(CID.findall(text)), "replacement": text.count("\ufffd")}


def needs_check(quality: dict, counts: dict) -> bool:
    """確かめるべきページか。判定（kind）は変えない。"""
    return quality["image_coverage"] >= CHECK_IMAGE_COVERAGE or counts["cid"] + counts["replacement"] >= 1


def _stable_group_textboxes(container, laparams, boxes):
    """pdfminer 20260107 の LTLayoutContainer.group_textboxes と同じ手順で、組の順だけを決まった番号で決める。

    元の関数は、ヒープの要素を (skip_isany, dist, id(obj1), id(obj2), obj1, obj2) とし、距離が同じ組の順を
    id(obj) で決める。ここでは id(obj) の代わりに、テキストボックスの並びの番号（0 から）と、まとめた組を
    作った順の番号を使う。番号はこの呼び出しの中だけで持つので、スレッドのあいだで状態を共有しない。
    """
    from pdfminer.layout import LTTextBoxVertical, LTTextGroupLRTB, LTTextGroupTBRL
    from pdfminer.utils import Plane

    plane = Plane(container.bbox)
    serial = {}

    def number(obj) -> int:
        key = id(obj)
        if key not in serial:
            serial[key] = (len(serial), obj)  # obj を持っておき、番地が使い回されないようにする
        return serial[key][0]

    def dist(obj1, obj2) -> float:
        x0 = min(obj1.x0, obj2.x0)
        y0 = min(obj1.y0, obj2.y0)
        x1 = max(obj1.x1, obj2.x1)
        y1 = max(obj1.y1, obj2.y1)
        return (x1 - x0) * (y1 - y0) - obj1.width * obj1.height - obj2.width * obj2.height

    def isany(obj1, obj2) -> set:
        x0 = min(obj1.x0, obj2.x0)
        y0 = min(obj1.y0, obj2.y0)
        x1 = max(obj1.x1, obj2.x1)
        y1 = max(obj1.y1, obj2.y1)
        return set(plane.find((x0, y0, x1, y1))).difference((obj1, obj2))

    for box in boxes:
        number(box)
    dists = []
    for i in range(len(boxes)):
        box1 = boxes[i]
        for j in range(i + 1, len(boxes)):
            box2 = boxes[j]
            dists.append((False, dist(box1, box2), number(box1), number(box2), box1, box2))
    heapq.heapify(dists)

    plane.extend(boxes)
    done = set()
    while dists:
        (skip_isany, d, id1, id2, obj1, obj2) = heapq.heappop(dists)
        if id1 not in done and id2 not in done:
            if not skip_isany and isany(obj1, obj2):
                heapq.heappush(dists, (True, d, id1, id2, obj1, obj2))
                continue
            if isinstance(obj1, (LTTextBoxVertical, LTTextGroupTBRL)) or \
                    isinstance(obj2, (LTTextBoxVertical, LTTextGroupTBRL)):
                group = LTTextGroupTBRL([obj1, obj2])
            else:
                group = LTTextGroupLRTB([obj1, obj2])
            plane.remove(obj1)
            plane.remove(obj2)
            done.update([id1, id2])
            for other in plane:
                heapq.heappush(dists, (False, dist(group, other), number(group), number(other), group, other))
            plane.add(group)
    return list(plane)


def _stable_converter(manager, output, laparams):
    """ページの組み立てに、組の順を決まった順にした LTPage を使う TextConverter を作る。

    LAParams の all_texts は既定（False）のままにする。図（LTFigure）の中の文字はまとめないので、
    group_textboxes を呼ぶのはページだけである。
    """
    from pdfminer.converter import TextConverter
    from pdfminer.layout import LTPage
    from pdfminer.utils import apply_matrix_rect

    class StablePage(LTPage):
        def group_textboxes(self, laparams, boxes):
            return _stable_group_textboxes(self, laparams, boxes)

    class StableTextConverter(TextConverter):
        def begin_page(self, page, ctm):
            (x0, y0, x1, y1) = apply_matrix_rect(ctm, page.mediabox)
            self.cur_item = StablePage(self.pageno, (0, 0, abs(x0 - x1), abs(y0 - y1)))

    return StableTextConverter(manager, output, codec="utf-8", laparams=laparams)


def pdfminer_pages(data: bytes, start: int = 1, stop: int | None = None) -> Iterator[tuple[int, str]]:
    """文書を 1 回だけ読み、ページの番号（1 から）と文字の層の本文を順に返す。

    start より前のページは読み飛ばす（解析しない）。stop を渡すと、そのページまでで止める。
    本文は改行を \\n にそろえ、前後の空白を除いたものである。
    """
    from pdfminer.layout import LAParams
    from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
    from pdfminer.pdfpage import PDFPage

    laparams = LAParams(**LAPARAMS)
    if laparams.all_texts:
        raise RuntimeError("all_texts を有効にすると、図の中の組の順が決まらない")
    output = io.StringIO()
    manager = PDFResourceManager()
    with _stable_converter(manager, output, laparams) as device:
        interpreter = PDFPageInterpreter(manager, device)
        for number, page in enumerate(PDFPage.get_pages(io.BytesIO(data)), start=1):
            if stop is not None and number > stop:
                return
            if number < start:
                continue
            interpreter.process_page(page)
            text = normalize_newlines(output.getvalue()).replace("\f", "").strip()
            output.seek(0)
            output.truncate(0)
            yield number, text


def pdfium_page_count(data: bytes) -> int:
    import pypdfium2 as pdfium

    with pdfium.PdfDocument(data) as document:
        return len(document)


def analyze_pages(data: bytes, start: int = 1, stop: int | None = None) -> Iterator[dict]:
    """ページごとに、判定、判定の数値、文字の層の本文、本文の崩れの数、確かめるべきかを返す。

    PDFium と pdfminer のページの数が違えば止まる（ページの対応が取れない）。
    """
    import pypdfium2 as pdfium

    with pdfium.PdfDocument(data) as document:
        count = len(document)
        last = count if stop is None else min(stop, count)
        seen = start - 1
        for number, text in pdfminer_pages(data, start, last):
            if number > count:
                break
            with closing(document[number - 1]) as page:
                _, quality = inspect_page(page)
            counts = garble_counts(text)
            seen = number
            yield {"page": number, "kind": quality["kind"], "quality": quality, "text": text,
                   "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                   "garble": counts, "needs_check": needs_check(quality, counts)}
        if seen != last:
            raise RuntimeError(f"PDFium と pdfminer のページの数が違う（PDFium {count}、pdfminer {seen} ページ目まで）")
    if stop is None or stop >= count:
        if sum(1 for _ in _pdfminer_page_objects(data)) != count:
            raise RuntimeError("PDFium と pdfminer のページの数が違う")


def _pdfminer_page_objects(data: bytes):
    from pdfminer.pdfpage import PDFPage

    return PDFPage.get_pages(io.BytesIO(data))


def page_ranges(numbers: list[int]) -> list[str]:
    """続いた番号を範囲にまとめる（[1, 2, 3, 5] → ["1-3", "5"]）。"""
    found: list[str] = []
    ordered = sorted(set(numbers))
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1] == ordered[j] + 1:
            j += 1
        found.append(str(ordered[i]) if i == j else f"{ordered[i]}-{ordered[j]}")
        i = j + 1
    return found


def page_list(numbers: list[int], start: int = 0, limit: int = LIST_LIMIT) -> dict:
    """ページの一覧の形（ADR-0025 の「ページの一覧の形」）。count は正確な数、ranges は start からの範囲。

    範囲の数が start + limit を超えるときは、続きを取る位置を next に入れる。続きがなければ next は None。
    """
    ranges = page_ranges(numbers)
    chosen = ranges[start:start + limit]
    following = start + limit if start + limit < len(ranges) else None
    return {"count": len(set(numbers)), "ranges": ",".join(chosen), "next": following}
