# info-hive-rag

RAG（Retrieval-Augmented Generation、検索拡張生成）を扱う、info-hive の Claude Code プラグインです。

プラグイン [wiki@info-hive](https://github.com/to884/info-hive-wiki) に依存します。`wiki` を有効にした状態で使います。

現在はプラグインの初期構成を用意した段階です。RAG の機能、スキル、コマンドは今後追加します。

## 名前

| 対象 | 名前 |
| --- | --- |
| リポジトリ | `to884/info-hive-rag` |
| プラグイン | `rag` |
| マーケットプレイス | `info-hive` |
| 導入するときの名前 | `rag@info-hive` |

## 導入

Claude Code で、マーケットプレイスを追加してから `rag` を入れます。依存先の `wiki` も自動で導入されます。

```text
/plugin marketplace add to884/info-hive-marketplace
/plugin install rag@info-hive
```

リポジトリは非公開のため、GitHub の認証情報が必要です。

## ライセンス

[MIT](LICENSE)
