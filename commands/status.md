---
description: PDF の索引件数と、失敗・中断した取り込みを確認する。
---

ユーザーの指定とプロジェクト・グローバルの指示から対象の wiki を特定し、`rag` MCP サーバーの `status` に `wiki_root` の絶対パスを渡す。

文書数、チャンク数、ジョブ ID、進捗と失敗理由を日本語で報告する。`queued`、`running`、`paused` は索引化完了ではない。次回 rag の MCP 接続時に未完了ジョブは自動再開する。失敗したジョブは理由を直して `retry_job(wiki_root, job_id)` で再試行する。キャンセルには `cancel_job` を使い、`cancellation_accepted` を確認する。公開開始後はキャンセルできない。キャンセル済みのジョブは明示的に再試行しない限り再開しない。
