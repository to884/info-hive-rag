"""OCR の確信度の低いページの確認（ADR-0025 の「OCR の確信度とページの確認」）。

確認の結果は wiki ごとに持つ。置き場所は 3 つに分ける。

- 抽出の結果（共有）: 文字の層の本文と OCR の結果。確認の結果は入れない。
- 確認の台帳（wiki ごと）: このモジュールの Ledger。wiki の作業領域の reviews.sqlite に置く。索引の台帳
  （catalog.sqlite）とは別のファイルにし、索引を空にしても（ADR-0014）消えない。取り込みの write.lock を
  待たずに確認を受け付けられる。
- 確認の結果を当てた本文と索引（wiki ごと）: rag_index.Engine が、台帳のある版までの確認を当てて作る。

このモジュールは標準ライブラリだけを使い、MCP のサーバーのプロセスからも読める。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import closing, contextmanager
from pathlib import Path

ACTIONS = ("accept", "fix", "reject")
DECIDED_BY = ("user", "agent_draft_approved")
# ページの状態。pending は確信度が低く、まだ決めていない（または決めた結果が古くなった）ページ
STATES = ("pending", "accepted", "fixed", "rejected", "stale")
REJECTED_LINE = "（OCR の結果を、確認で却下した）"


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Ledger:
    """wiki ごとの確認の台帳。版は、wiki と PDF ごとに、確認を受け付けるたびに 1 つ上がる。前の結果は消さない。"""

    def __init__(self, folder: Path):
        self.path = Path(folder) / "reviews.sqlite"

    @contextmanager
    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=15)) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA busy_timeout=15000")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS review_versions (
                    doc_id TEXT PRIMARY KEY, version INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS review_decisions (
                    doc_id TEXT NOT NULL, version INTEGER NOT NULL, page INTEGER NOT NULL,
                    action TEXT NOT NULL, text TEXT, decided_by TEXT NOT NULL, ocr_sha256 TEXT NOT NULL,
                    decided_at REAL NOT NULL, PRIMARY KEY (doc_id, version, page));
            """)
            with db:
                yield db

    def version(self, doc_id: str) -> int:
        if not self.path.exists():
            return 0
        with self.connect() as db:
            row = db.execute("SELECT version FROM review_versions WHERE doc_id=?", (doc_id,)).fetchone()
        return int(row[0]) if row else 0

    def decisions(self, doc_id: str, upto: int | None = None) -> dict[int, dict]:
        """ページごとの、upto の版までの最後の結果を返す。upto を省くと台帳の最新の版まで。"""
        if not self.path.exists():
            return {}
        with self.connect() as db:
            rows = db.execute("""SELECT * FROM review_decisions WHERE doc_id=? AND (? IS NULL OR version<=?)
                ORDER BY version""", (doc_id, upto, upto)).fetchall()
        found: dict[int, dict] = {}
        for row in rows:
            found[int(row["page"])] = dict(row)
        return found

    def history(self, doc_id: str, page: int) -> list[dict]:
        if not self.path.exists():
            return []
        with self.connect() as db:
            return [dict(row) for row in db.execute(
                "SELECT * FROM review_decisions WHERE doc_id=? AND page=? ORDER BY version", (doc_id, page))]

    def decide(self, doc_id: str, base_version: int, items: list[dict], current: dict[int, str]) -> int:
        """確認の結果を受け付け、新しい台帳の版を返す。

        base_version が台帳の版と違うか、確かめたときの OCR の本文の SHA-256（ocr_sha256）が今の OCR の本文
        （current: ページ → SHA-256）と違えば、受け付けない。版の照合と版上げは、1 つのトランザクションで行う。
        """
        if not items:
            raise ValueError("確認の結果が空です。")
        pages = set()
        for item in items:
            page = int(item.get("page", 0))
            if page in pages:
                raise ValueError(f"{page} ページ目の結果が 2 つあります。")
            pages.add(page)
            if page not in current:
                raise ValueError(f"{page} ページ目は、この PDF の OCR したページではありません。")
            if item.get("action") not in ACTIONS:
                raise ValueError("action は accept / fix / reject にしてください。")
            if item.get("decided_by") not in DECIDED_BY:
                raise ValueError("decided_by は user / agent_draft_approved にしてください。")
            if item["action"] == "fix" and not str(item.get("text") or "").strip():
                raise ValueError(f"{page} ページ目を直すときは、直した本文（text）を渡してください。")
            if item.get("ocr_sha256") != current[page]:
                raise ValueError(f"{page} ページ目の OCR の本文が、確かめたときと違います。ocr_page_detail で取り直してください。")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT version FROM review_versions WHERE doc_id=?", (doc_id,)).fetchone()
            now_version = int(row[0]) if row else 0
            if now_version != int(base_version):
                raise ValueError(f"台帳の版が {now_version} になっています（もとにした版は {base_version}）。"
                                 "別のセッションで決めた結果を上書きしないよう、ocr_page_detail で取り直してください。")
            version = now_version + 1
            now = time.time()
            for item in items:
                db.execute("INSERT INTO review_decisions VALUES (?,?,?,?,?,?,?,?)",
                           (doc_id, version, int(item["page"]), item["action"],
                            item.get("text") if item["action"] == "fix" else None,
                            item["decided_by"], item["ocr_sha256"], now))
            db.execute("INSERT INTO review_versions VALUES (?,?) ON CONFLICT(doc_id) DO UPDATE SET version=excluded.version",
                       (doc_id, version))
        return version


def is_low(entry: dict, threshold: float) -> bool | None:
    """OCR したページの確信度が低いか。確信度が不明なら None を返す（低いとも高いとも扱わない）。"""
    minimum = entry.get("confidence_min")
    if minimum is None:
        return None
    return minimum < threshold


def page_state(entry: dict, decision: dict | None, threshold: float) -> str | None:
    """OCR したページの確認の状態。OCR していないページと、確信度が高いか不明で決めていないページは None。"""
    if not entry.get("ocr"):
        return None
    if decision is not None:
        if decision["ocr_sha256"] != entry.get("ocr_sha256"):
            return "stale"  # OCR をやり直して本文が変わった。決めた結果は当てず、確認待ちに戻す
        return {"accept": "accepted", "fix": "fixed", "reject": "rejected"}[decision["action"]]
    return "pending" if is_low(entry, threshold) else None


def apply_decisions(pages: list[dict], decisions: dict[int, dict]) -> list[dict]:
    """抽出の結果のページに確認の結果を当てた、新しいページの一覧を返す（元のページは変えない）。

    直したページは直した本文に、却下したページは却下を書いた 1 行にする（索引には入れない）。古くなった
    結果（OCR の本文の SHA-256 が違う）は当てない。
    """
    applied = []
    for page in pages:
        decision = decisions.get(page["page"])
        if decision is None or not page.get("ocr") or decision["ocr_sha256"] != page.get("ocr_sha256"):
            applied.append(page)
        elif decision["action"] == "fix":
            applied.append({**page, "text": decision["text"], "reviewed": "fixed"})
        elif decision["action"] == "reject":
            applied.append({**page, "text": REJECTED_LINE, "reviewed": "rejected"})
        else:
            applied.append({**page, "reviewed": "accepted"})
    return applied


def confidence_summary(pages: list[dict]) -> dict:
    """wiki の作業領域に置く、文書ごとのページの確信度の要約（検索の印と status の数で使う）。"""
    return {"pages": {str(page["page"]): {"ocr": bool(page.get("ocr")), "ocr_sha256": page.get("ocr_sha256"),
                                          "confidence_min": page.get("confidence_min")}
                      for page in pages if page.get("ocr")}}


def load_summary(folder: Path, doc_id: str) -> dict[int, dict]:
    path = Path(folder) / "reviewed" / doc_id / "confidence.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(page): entry for page, entry in data["pages"].items()}


def review_lists(summary: dict[int, dict], ledger_decisions: dict[int, dict], published_decisions: dict[int, dict],
                 threshold: float) -> dict[str, list[int]]:
    """確認の状態ごとのページ。公開した版の状態から作り、台帳の版が新しいページは awaiting_publish にも入れる。"""
    lists: dict[str, list[int]] = {f"review:{state}": [] for state in STATES}
    lists["review:awaiting_publish"] = []
    lists["ocr_low_confidence"] = []
    lists["ocr_unknown_confidence"] = []
    for page, entry in sorted(summary.items()):
        low = is_low(entry, threshold)
        if low:
            lists["ocr_low_confidence"].append(page)
        elif low is None:
            lists["ocr_unknown_confidence"].append(page)
        state = page_state(entry, published_decisions.get(page), threshold)
        if state:
            lists[f"review:{state}"].append(page)
        latest = ledger_decisions.get(page)
        published = published_decisions.get(page)
        if latest is not None and (published is None or latest["version"] != published["version"]):
            lists["review:awaiting_publish"].append(page)
    return lists


def published_state(folder: Path, doc_id: str, fingerprint: str) -> int:
    """その wiki の索引で公開している確認の版。文書が索引にないか、方針の違う索引なら止まる。"""
    catalog = Path(folder) / "catalog.sqlite"
    row = None
    if catalog.exists():
        with closing(sqlite3.connect(catalog, timeout=15)) as db:
            columns = {item[1] for item in db.execute("PRAGMA table_info(documents)")}
            if "review_version" in columns:
                row = db.execute("SELECT review_version,fingerprint FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
    if row is None:
        raise ValueError("この wiki の索引に、指定の PDF がありません。PDF を取り込んでください。")
    if row[1] != fingerprint:
        raise ValueError("索引の方針が今の設定と違います。PDF を取り込み直してから確かめてください。")
    return int(row[0])


def ocr_pages(config, doc_id: str) -> dict[int, dict]:
    """今の設定の抽出の結果から、OCR したページの記録を返す。"""
    cache = config.extraction_dir(doc_id)
    found = {}
    for path in cache.glob("page-*.json"):
        entry = json.loads(path.read_text(encoding="utf-8"))
        if entry.get("ocr") and entry.get("extraction_key") == config.extraction_key:
            found[int(entry["page"])] = entry
    return found


def page_detail(config, wiki: str, doc_id: str, page: int) -> dict:
    """ocr_page_detail の結果。OCR の本文は共有の抽出の結果から返すので、直したり却下したりしたあとでも元の本文が分かる。"""
    from rag_pdf import OcrStore
    folder = config.workspace(wiki)
    published = published_state(folder, doc_id, config.fingerprint)
    entry = ocr_pages(config, doc_id).get(int(page))
    if entry is None:
        raise ValueError(f"{page} ページ目は、この PDF の OCR したページではありません。")
    stored = OcrStore(config, doc_id).load(int(page))
    record = (stored or {}).get("confidence")
    threshold = config.ocr_confidence_threshold
    lines = None
    if record:
        lines = [{"line": number, "text": line["text"], "confidence": line["confidence"], "box": line.get("box"),
                  "vertical": line.get("vertical")}
                 for number, line in enumerate(record["lines"], start=1) if line["confidence"] < threshold]
    ledger = Ledger(folder)
    return {"page": int(page), "ocr_text": entry["text"], "ocr_sha256": entry["ocr_sha256"],
            "confidence_min": entry.get("confidence_min"), "confidence_known": entry.get("confidence_min") is not None,
            "threshold": threshold, "low_lines": lines,
            "state": page_state(entry, ledger.decisions(doc_id, upto=published).get(int(page)), threshold),
            "decision": ledger.decisions(doc_id).get(int(page)), "history": ledger.history(doc_id, int(page)),
            "ledger_version": ledger.version(doc_id), "published_version": published}


def decide(config, wiki: str, doc_id: str, base_version: int, items: list[dict]) -> int:
    """decide_ocr_pages の本体。台帳の版を返す。作り直しのジョブの登録は呼ぶ側が行う。"""
    folder = config.workspace(wiki)
    published_state(folder, doc_id, config.fingerprint)
    current = {page: entry["ocr_sha256"] for page, entry in ocr_pages(config, doc_id).items()}
    return Ledger(folder).decide(doc_id, base_version, items, current)
