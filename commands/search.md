---
description: wiki に取り込んだ PDF を、ページと出典付きで検索する。
argument-hint: "<質問>"
---

ユーザーが指定した wiki、プロジェクトの指示、グローバルな指示、現在のプロジェクトの `wiki/` の順に `<WIKI_ROOT>` を決める。

`rag` MCP サーバーの `search` を、`wiki_root=<WIKI_ROOT の絶対パス>`、`query=$ARGUMENTS` で呼ぶ。
結果の本文とページ番号、`raw`、`source_pages` を根拠に日本語で答える。結果にない事実を補わず、根拠が足りなければ追加の検索を行う。回答の生成は Claude Code が行う。
