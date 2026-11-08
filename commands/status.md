---
description: PDF の索引件数と、失敗・中断した取り込みを確認する。
---

ユーザーの指定とプロジェクト・グローバルの指示から対象の wiki を特定し、`rag` MCP サーバーの `status` に `wiki_root` の絶対パスを渡す。

文書数、チャンク数、ジョブ ID、進捗と失敗理由を日本語で報告する。`queued`、`running`、`paused` は索引化完了ではない。次回 rag の MCP 接続時に未完了ジョブは自動再開する。失敗したジョブは理由を直して `retry_job(wiki_root, job_id)` で再試行する。キャンセルには `cancel_job` を使い、`cancellation_accepted` を確認する。公開開始後はキャンセルできない。キャンセル済みのジョブは明示的に再試行しない限り再開しない。

応答に `legacy_data` があれば、rag 0.3.0 のデータ（`paths` の場所）が残っていることを伝える。新しい版はそれを読まないので、ユーザーが消してよい。消すのはユーザーに任せ、自分では消さない。0.3.0 で取り込んだ PDF は、`/wiki:ingest` で取り込み直す。

応答の `reingest_required` が `true` なら、埋め込みモデルや OCR の設定が変わったので、`reingest` の PDF を取り込み直す必要がある。1 件も取り込み直していないうちは、その wiki を検索できない。取り込み直し始めたら、取り込み直した PDF だけを検索できる。`reingest` の各項目の `paths`（PDF の場所）、`title`、`source_pages` を示し、`/wiki:ingest` で取り込み直すよう伝える。抽出の設定が前と同じなら、取り込み直しで OCR はやり直さない。
