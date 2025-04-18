# Contributing

PAMIQ Core への貢献に興味を持っていただき、ありがとうございます。このガイドでは開発環境のセットアップ方法について説明します。

## 📋 前提条件

以下のツールを事前にインストールしてください:

### 必須ツール

- 🐳 **Docker (Docker Compose)**

    - Docker Desktop: <https://www.docker.com/ja-jp/get-started/>
    - Docker Engine (Linux 限定): <https://docs.docker.com/engine/install/>
    - 確認コマンド:
        ```sh
        docker version && docker compose version
        ```

- 🔨 **make**

    - Windows: [`scoop`](https://scoop.sh)か[`chocolate`](https://chocolatey.org)でインストール
    - macOS: 事前にインストール済み
    - Linux: ディストリビューションのパッケージマネージャーを使用（例：Ubuntu の場合 `sudo apt install make`）
    - 確認コマンド:
        ```sh
        make -v
        ```

- 🌲 **git**

    - ダウンロード: <https://git-scm.com/downloads>
    - 確認コマンド:
        ```sh
        git -v
        ```

## 🚀 開発環境の構築

1. リポジトリのセットアップ

    ```sh
    git clone https://github.com/MLShukai/pamiq-core.git
    cd pamiq-core
    ```

2. Docker環境の構築

    ```sh
    # イメージのビルド
    make docker-build

    # コンテナの起動
    make docker-up

    # コンテナへの接続
    make docker-attach
    ```

3. Gitの初期設定

    ```sh
    git config user.name <あなたのGitHubユーザー名>
    git config user.email <あなたのGitHubメールアドレス>
    ```

## 💻 開発環境の設定

### VSCode での開発

お好みのエディタ（VSCode 推奨）からコンテナにアタッチして開発を行えます。

📚 参考: [VSCode Dev Containers 拡張機能でアタッチ](https://code.visualstudio.com/docs/devcontainers/attach-container)

開発コンテナには以下の環境が整っています：

- パッケージマネージャー ([**uv**](https://docs.astral.sh/uv/))
- バージョン管理用のGit
- 開発の依存関係パッケージ
- 仮想環境の自動アクティベーション (`source .venv/bin/activate`)

## 🔄 開発ワークフロー

以下のコマンドで開発を行います：

```sh
# Python仮想環境の構築
make venv

# コードのフォーマットとpre-commitフックの実行
make format

# テストの実行
make test

# 型チェックの実行
make type

# 全ワークフローの実行（format、test、type）
make run
```

## ⚙️ 環境の管理

### コンテナの停止

```sh
make docker-down
```

### 開発環境のクリーンアップ

```sh
make clean
```

### ⚠️ 完全削除（要注意）

```sh
# 警告: 全ての作業データが削除されます！
make docker-down-volume
```

## 🤝 コントリビューションの流れ

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
