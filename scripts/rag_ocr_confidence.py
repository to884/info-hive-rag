"""NDLOCR-Lite の認識器（PARSeq）の出力から、文字ごとの確率を数える（ADR-0025 の「確信度の数え方」）。

NDLOCR-Lite のファイルは変えない。PARSEQ.read と同じ前処理と同じ選び方（argmax、番号 0 の終わりの記号で止める）で
読み、ロジットに softmax をかけた確率も受け取る。読んだ文字列は ProbStr（str の子クラス）にして確率を持たせる。
NDLOCR-Lite の process_cascade は、読みの文字列の長さで別の認識器に読み直させ、長い行は左右に分けて読み、
文字列を + でつなぐ。ProbStr は + でも確率をつなぐので、process_cascade をそのまま使える。

rag_ocr.py（OCR の環境）から読み込む。numpy だけを使う。
"""

from __future__ import annotations

import numpy as np

# rag_config.OCR_CONFIDENCE_REVISION と同じ値にする（OCR の環境は rag_config を読み込まない）
CONFIDENCE_REVISION = "confidence-v1"
DIGITS = 4


class ProbStr(str):
    """確率を持つ読みの文字列。chars は文字ごとの確率、eos は読みごとの終わりの記号の確率、reads は読みの数。"""

    def __new__(cls, text: str, chars, eos, reads: int):
        value = super().__new__(cls, text)
        value.chars = [float(p) for p in chars]
        value.eos = [float(p) for p in eos]
        value.reads = int(reads)
        return value

    def __add__(self, other):
        if not isinstance(other, ProbStr):
            return str(self) + other  # 確率のない文字列とつないだら、確率は持たない（確信度が不明になる）
        return ProbStr(str(self) + str(other), self.chars + other.chars, self.eos + other.eos, self.reads + other.reads)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class ConfidentRecognizer:
    """PARSEQ を包み、read が ProbStr を返すようにする。呼ぶたびに model.session を読む（作り直したセッションを使う）。"""

    def __init__(self, model):
        self.model = model

    def read(self, img) -> str:
        model = self.model
        if img is None or img.size == 0:
            return ProbStr("", [], [], 0)  # 読んでいない（PARSEQ.read も空の文字列を返す）
        tensor = model.preprocess(img)
        outputs = model.session.run(model.output_names, {model.input_names[0]: tensor})[0]
        logits = outputs[0]
        indices = np.argmax(logits, axis=1)
        stop = np.where(indices == 0)[0]
        end = int(stop[0]) if stop.size > 0 else len(indices)
        chosen = indices[:end].tolist()
        text = "".join(model.charlist[i - 1] for i in chosen)
        probabilities = softmax(logits.astype(np.float64))
        chars = probabilities[np.arange(end), indices[:end]] if end else np.zeros(0)
        # 最大の長さに達して終わりの記号がない読みは、終わりの記号の確率を数えない
        eos = [probabilities[end, 0]] if end < len(indices) else []
        return ProbStr(text, chars, eos, 1)


def line_records(json_lines: list[dict]) -> list[dict]:
    """NDLOCR-Lite の json_lines から、保存する行の記録を作る。

    行の確信度は、文字の確率と終わりの記号の確率の最小値。認識した本文が空の行は「読めなかった行」として 0 にする。
    確率のない行（ProbStr でない）と、不変条件（文字の確率の数が本文の文字の数と同じ、終わりの記号の確率の数が
    読みの数以下）を満たさない行は、確信度を None にする（そのページは確信度が不明になる）。
    """
    records = []
    for line in json_lines:
        text = line.get("text", "")
        (x0, y0), _, (x1, _), (_, y1) = line["boundingBox"]
        record = {"text": str(text), "box": [x0, y0, x1 - x0, y1 - y0], "vertical": line.get("isVertical") == "true",
                  "detector_confidence": round(float(line.get("confidence", 0.0)), DIGITS)}
        if not isinstance(text, ProbStr):
            record.update(chars=None, eos=None, confidence=None)
        elif len(text.chars) != len(str(text)) or len(text.eos) > text.reads:
            record.update(chars=None, eos=None, confidence=None, invariant="broken")
        elif not str(text):
            record.update(chars=[], eos=[round(p, DIGITS) for p in text.eos], confidence=0.0, unreadable=True)
        else:
            values = text.chars + text.eos
            record.update(chars=[round(p, DIGITS) for p in text.chars], eos=[round(p, DIGITS) for p in text.eos],
                          confidence=round(min(values), DIGITS))
        records.append(record)
    return records
