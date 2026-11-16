"""OCR の行の印（ADR-0025 の「行の印」）。2 つ目の読みと比べて、誤っている見込みの高い行に印を付ける。

印の理由は 4 つある。
- "low": 行の確信度が閾値（ocr_confidence_threshold）より低い。閾値は設定で変わるので、保存せず、数えるときに決める。
- "blur": 行の画像を少しぼかして読み直した本文（OCR の環境の rag_ocr が返す blur_text）が、採った本文と違う。
- "alt": ページを別の解像度で描き直し、検出から読み直した本文が、採った本文と違い、かつ行の確信度が 0.9 未満。
- "rapid": 横書きの行の画像を、別の認識器（RapidOCR の日本語の認識モデル）で読んだ本文（rapid_text）が、採った本文に
  ない英数字か記号を足すか置き換えた。NDLOCR-Lite の 3 つの認識器がそろって落とす文字（ローマ数字、注の番号など）を捕える。
  RapidOCR が文字を落としただけの違い（RapidOCR の文字の一覧にない記号）、見た目の同じ文字の違い、取り違えやすい英数字
  どうし（O と 0 と o、l と 1 と I と |）の入れ替えは数えない（RapidOCR 自身がよく読み違える）。

"blur"、"alt"、"rapid" は、OCR したときに決まるので、OCR の結果と一緒に保存する（static_flags）。
本文の比べ方と、行の対応と、読む順は、承認の条件 5 の測定（docs/research/adr-0025-condition5-2026-11-16/scripts/score.py）と
同じにする。依存は標準ライブラリだけにする。
"""

from __future__ import annotations

import unicodedata

# 印の決め方を変えたら上げる（保存した印は、版が違えば「不明」として扱う）
OCR_FLAG_REVISION = "flags-v2"
# 別の解像度の食い違いを印にするのは、行の確信度がこの値より低いときだけ
ALT_CONFIDENCE_LIMIT = 0.9
# 見た目の同じ記号をそろえる表（測定と同じ）
SAME = {**{c: "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uff0d"}, "\u301c": "~", "\uff5e": "~",
        **{c: "'" for c in "\u2018\u2019\u201a\u201b"}, **{c: '"' for c in "\u201c\u201d\u201e\u201f"},
        "\uff65": "\u30fb", "\u00b7": "\u30fb", "\u25cb": "\u3007"}


# 2 つの認識器の読みを比べるときだけ、見た目の同じ文字をそろえる（カタカナと漢字、長音とダッシュ）
LOOKALIKE = {"卜": "ト", "口": "ロ", "力": "カ", "工": "エ", "夕": "タ", "二": "ニ", "八": "ハ", "才": "オ",
             "一": "ー", "-": "ー", "へ": "ヘ", "べ": "ベ", "ぺ": "ペ", "り": "リ"}
# 取り違えやすい英数字の組。この組の中だけの 1 文字ずつの入れ替えは、"rapid" の印にしない
AMBIGUOUS = (frozenset("O0o"), frozenset("l1I|"))


def alt_dpi(ocr_dpi: int) -> int:
    """読み直しに使う解像度（ocr_dpi の 4/3 倍。既定の 150 なら 200）。"""
    return round(ocr_dpi * 4 / 3)


def normalize(text: str) -> str:
    """比べるための形にする。NFKC、空白を除く、見た目の同じ記号をそろえる。"""
    text = unicodedata.normalize("NFKC", text)
    return "".join(SAME.get(c, c) for c in text if not c.isspace())


def corners(box) -> list[float]:
    """行の記録の箱 [x, y, 幅, 高さ] を [x0, y0, x1, y1] にする。"""
    x, y, w, h = box
    return [x, y, x + w, y + h]


def area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def joined(a, b) -> bool:
    """2 つの箱（[x0, y0, x1, y1]）の重なりの面積が、小さい方の箱の面積の半分以上なら、つながる。"""
    inter = [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]
    smaller = min(area(a), area(b))
    return smaller > 0 and area(inter) >= 0.5 * smaller


def reading_order(lines: list[dict], vertical: bool) -> list[dict]:
    """読む順に並べる（lines の各要素は "box" に [x0, y0, x1, y1] を持つ）。

    横書きは上から下、同じ高さは左から右。縦書きは右から左、同じ列は上から下。入力の順によらない全順序にするため、
    先に行を帯（横書きは高さの帯、縦書きは列）にまとめる。行の中心の順に見ていき、今の帯の範囲と、行の高さ
    （縦書きは幅）の半分以上が重なれば同じ帯に入れ、帯の範囲を広げる。同じ値は、箱と本文で決める。
    """
    if vertical:
        def across(b):
            return (-b[2], -b[0])

        def along(b):
            return b[1]
    else:
        def across(b):
            return (b[1], b[3])

        def along(b):
            return b[0]

    def tie(line):
        return (tuple(line["box"]), str(line.get("text", "")))

    ordered = sorted(lines, key=lambda x: ((across(x["box"])[0] + across(x["box"])[1]) / 2, along(x["box"]), tie(x)))
    bands: list[list] = []
    for line in ordered:
        lo, hi = across(line["box"])
        if bands:
            band_lo, band_hi, members = bands[-1]
            inter = min(hi, band_hi) - max(lo, band_lo)
            if inter >= 0.5 * max(1e-9, min(hi - lo, band_hi - band_lo)):
                bands[-1] = [min(lo, band_lo), max(hi, band_hi), members + [line]]
                continue
        bands.append([lo, hi, [line]])
    return [line for _, _, members in bands for line in sorted(members, key=lambda x: (along(x["box"]), tie(x)))]


def alt_texts(lines: list[dict], alt_lines: list[dict], scale: float) -> list[str]:
    """採った行ごとに、別の解像度で読んだ行のうちつながる行を、読む順につないだ本文を返す。

    alt_lines の箱は scale 倍して、採った行の座標にそろえる。つながる行がなければ空の本文にする。
    """
    others = [{"text": x.get("text", ""), "box": [v * scale for v in corners(x["box"])]} for x in alt_lines]
    result = []
    for line in lines:
        box = corners(line["box"])
        matched = [o for o in others if joined(o["box"], box)]
        result.append("".join(o["text"] for o in reading_order(matched, bool(line.get("vertical")))))
    return result


def skeleton(text: str) -> str:
    """2 つの認識器の読みを比べる形。normalize のあと、見た目の同じ文字をそろえる。"""
    return "".join(LOOKALIKE.get(c, c) for c in normalize(text))


def rapid_added(text: str, rapid_text: str) -> bool:
    """別の認識器の読みが、採った本文にない英数字か記号を足すか置き換えたか（"rapid" の印の決まり）。"""
    import difflib

    a, b = skeleton(text), skeleton(rapid_text)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag in ("equal", "delete"):
            continue
        old, new = a[i1:i2], b[j1:j2]
        if tag == "replace" and len(old) == len(new) and all(
                any(x in group and y in group for group in AMBIGUOUS) for x, y in zip(old, new)):
            continue
        if any((c.isascii() and c.isalnum()) or not c.isalnum() for c in new + old):
            return True
    return False


def static_flags(line: dict) -> list[str]:
    """OCR したときに決まる印の理由（"blur"、"alt"、"rapid"）。line は blur_text、alt_text、rapid_text を持つ
    （rapid_text は、縦書きの行では None）。"""
    reasons = []
    text = normalize(line.get("text", ""))
    if line.get("blur_text") is not None and normalize(line["blur_text"]) != text:
        reasons.append("blur")
    confidence = line.get("confidence")
    if (line.get("alt_text") is not None and normalize(line["alt_text"]) != text
            and confidence is not None and confidence < ALT_CONFIDENCE_LIMIT):
        reasons.append("alt")
    if line.get("rapid_text") is not None and rapid_added(line.get("text", ""), line["rapid_text"]):
        reasons.append("rapid")
    return reasons


def reasons_for(line: dict, threshold: float) -> list[str]:
    """印の理由をすべて返す（"low" は、そのときの閾値で決める）。空なら印はない。"""
    reasons = []
    if line.get("confidence") is not None and line["confidence"] < threshold:
        reasons.append("low")
    return reasons + list(line.get("flags") or [])
