# Contributing

PAMIQ Core への貢献に興味を持っていただき、ありがとうございます。このガイドでは開発環境のセットアップ方法について説明します。

## セットアップ

### 前提条件

- [**Docker**](https://www.docker.com/ja-jp/get-started/)
- **make**
  - Windows ユーザーは [**chocolatey**](https://chocolatey.org) または [**scoop**](https://scoop.sh) を使用してインストールしてください
  - macOS ユーザーは事前にインストールされています
  - Linux ユーザーはディストリビューションのパッケージマネージャーを使用してインストールしてください（例：Ubuntu の場合 `sudo apt install make`）
- [**git**](https://git-scm.com)

### 開発環境の構築

1. リポジトリをクローン

   ```shell
   git clone https://github.com/your-username/pamiq-core.git
   cd pamiq-core
   ```

2. Docker環境を起動

   ```shell
   make docker-up
   ```

3. 開発コンテナにアタッチ

   ```shell
   make docker-attach
   ```

   Visual Studio Codeなどを用いて開発コンテナにアタッチして作業することを推奨します。

4. Gitのユーザ名、Emailアドレスを登録

   ```shell
   git config user.name <your-username>
   git config user.email <your-email>
   ```

開発コンテナには以下の環境が整っています：

- パッケージマネージャー ([**uv**](https://docs.astral.sh/uv/))
- バージョン管理用のGit
- 必要な開発依存関係
- 仮想環境の自動アクティベーション

## 開発ワークフロー

以下のコマンドで開発を行います：

```sh
# コードのフォーマットとpre-commitフックの実行
make format

# テストの実行
make test

# 型チェックの実行
make type

# 全ワークフローの実行（format、test、type）
make run
```

## 開発環境のクリーンアップ

不要なファイルやキャッシュを削除する場合：

```sh
make clean
```

## コントリビューションの流れ

1. 機能追加やバグ修正用の新しいブランチを作成
2. 変更を加える
3. 新機能のテストを記述
4. PRを送る前に全ワークフローを実行：
   ```shell
   make run
   ```
5. 変更内容を明確に説明したPull Requestを提出

質問や問題がある場合は、GitHubリポジトリでIssueを作成してください。

[また詳しい開発の情報はWikiを参照してください。](https://github.com/MLShukai/pamiq-core/wiki)
