```bash
#!/bin/bash

# カラー設定
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ログ関数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# エラーハンドリング
set -e
trap 'log_error "セットアップに失敗しました。エラーコード: $?"' ERR

# ヘルプメッセージ
show_help() {
    echo "AI-Driven Operations Platform セットアップスクリプト"
    echo
    echo "使用方法: $0 [オプション]"
    echo
    echo "オプション:"
    echo "  -h, --help                 ヘルプメッセージを表示"
    echo "  -e, --environment [ENV]    環境を指定 (development/staging/production)"
    echo "  --skip-docker             Dockerのインストールをスキップ"
    echo "  --skip-poetry             Poetryのインストールをスキップ"
    echo "  --skip-tools              追加ツールのインストールをスキップ"
    echo
}

# デフォルト設定
ENVIRONMENT="development"
SKIP_DOCKER=false
SKIP_POETRY=false
SKIP_TOOLS=false

# 引数のパース
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --skip-docker)
            SKIP_DOCKER=true
            shift
            ;;
        --skip-poetry)
            SKIP_POETRY=true
            shift
            ;;
        --skip-tools)
            SKIP_TOOLS=true
            shift
            ;;
        *)
            log_error "不明なオプション: $1"
            show_help
            exit 1
            ;;
    esac
done

# 必要なディレクトリの作成
create_directories() {
    log_info "必要なディレクトリを作成中..."
    
    directories=(
        "logs"
        "data"
        "config"
        "models"
        "temp"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        log_info "作成: $dir/"
    done
}

# Pythonの依存関係チェック
check_python() {
    log_info "Pythonの依存関係をチェック中..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 が見つかりません"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -V 2>&1 | grep -Po '(?<=Python )(.+)')
    if [[ $(echo "$PYTHON_VERSION 3.9" | tr " " "\n" | sort -V | head -n 1) != "3.9" ]]; then
        log_error "Python 3.9以上が必要です。現在のバージョン: $PYTHON_VERSION"
        exit 1
    fi
    
    log_info "Python $PYTHON_VERSION が見つかりました"
}

# Dockerのインストール
install_docker() {
    if [ "$SKIP_DOCKER" = true ]; then
        log_warning "Dockerのインストールをスキップします"
        return
    fi
    
    log_info "Dockerをインストール中..."
    
    if command -v docker &> /dev/null; then
        log_info "Docker は既にインストールされています"
    else
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker $USER
        rm get-docker.sh
        
        log_info "Docker のインストールが完了しました"
    fi
    
    # Docker Composeのインストール
    if ! command -v docker-compose &> /dev/null; then
        log_info "Docker Composeをインストール中..."
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
    fi
}

# Poetryのインストール
install_poetry() {
    if [ "$SKIP_POETRY" = true ]; then
        log_warning "Poetryのインストールをスキップします"
        return
    fi
    
    log_info "Poetryをインストール中..."
    
    if command -v poetry &> /dev/null; then
        log_info "Poetry は既にインストールされています"
    else
        curl -sSL https://install.python-poetry.org | python3 -
        
        # PATHの設定
        export PATH="$HOME/.local/bin:$PATH"
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
        
        log_info "Poetry のインストールが完了しました"
    fi
    
    # Poetry の設定
    poetry config virtualenvs.in-project true
}

# 追加ツールのインストール
install_tools() {
    if [ "$SKIP_TOOLS" = true ]; then
        log_warning "追加ツールのインストールをスキップします"
        return
    fi
    
    log_info "追加ツールをインストール中..."
    
    # 必要なパッケージのインストール
    sudo apt-get update
    sudo apt-get install -y \
        git \
        make \
        curl \
        jq \
        unzip \
        htop \
        vim
    
    # Terraformのインストール
    if ! command -v terraform &> /dev/null; then
        log_info "Terraformをインストール中..."
        curl -fsSL https://apt.releases.hashicorp.com/gpg | sudo apt-key add -
        sudo apt-add-repository "deb [arch=amd64] https://apt.releases.hashicorp.com $(lsb_release -cs) main"
        sudo apt-get update
        sudo apt-get install -y terraform
    fi
    
    # Ansibleのインストール
    if ! command -v ansible &> /dev/null; then
        log_info "Ansibleをインストール中..."
        sudo apt-get install -y ansible
    fi
}

# 環境設定ファイルの生成
generate_env_file() {
    log_info "環境設定ファイルを生成中..."
    
    if [ -f .env ]; then
        log_warning "既存の .env ファイルをバックアップします"
        mv .env .env.backup
    fi
    
    cat > .env << EOL
# 基本設定
ENVIRONMENT=${ENVIRONMENT}
APP_NAME=ai-ops-platform
DEBUG=$([ "$ENVIRONMENT" = "development" ] && echo "true" || echo "false")
SECRET_KEY=$(openssl rand -hex 32)

# サーバー設定
HOST=0.0.0.0
PORT=8000
WORKERS=4

# データベース設定
DB_HOST=localhost
DB_PORT=5432
DB_NAME=aiops
DB_USER=admin
DB_PASSWORD=$(openssl rand -hex 16)

# Redis設定
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=$(openssl rand -hex 16)

# Elasticsearch設定
ELASTICSEARCH_HOSTS=http://localhost:9200
ELASTICSEARCH_INDEX_PREFIX=aiops-

# Prometheus設定
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090

# JWT設定
JWT_SECRET=$(openssl rand -hex 32)
JWT_ALGORITHM=HS256
JWT_EXPIRATION=3600

# ログ設定
LOG_LEVEL=$([ "$ENVIRONMENT" = "development" ] && echo "DEBUG" || echo "INFO")
LOG_FORMAT=json

# AWS設定
AWS_REGION=us-west-2
EOL

    log_info ".env ファイルを生成しました"
}

# メイン処理
main() {
    log_info "セットアップを開始します: $ENVIRONMENT 環境"
    
    # 依存関係のチェックとインストール
    check_python
    create_directories
    install_docker
    install_poetry
    install_tools
    
    # 環境設定
    generate_env_file
    
    # Poetry依存関係のインストール
    log_info "Python依存関係をインストール中..."
    poetry install
    
    # 完了メッセージ
    log_info "セットアップが完了しました!"
    log_info "以下のコマンドで開発サーバーを起動できます:"
    echo "  poetry run python -m src.main"
}

# スクリプトの実行
main

exit 0
```

このセットアップスクリプトは以下の機能を提供します：

1. **環境のセットアップ**
   - 必要なディレクトリの作成
   - Python依存関係のチェック
   - Dockerのインストール
   - Poetryのインストール
   - 追加ツールのインストール

2. **設定の生成**
   - 環境設定ファイルの生成
   - シークレットキーの生成
   - 環境に応じた設定

3. **ユーティリティ**
   - カラー付きログ出力
   - エラーハンドリング
   - ヘルプメッセージ

使用方法：

1. **基本的な使用**
```bash
./scripts/setup.sh
```

2. **環境の指定**
```bash
./scripts/setup.sh -e production
```

3. **特定のインストールをスキップ**
```bash
./scripts/setup.sh --skip-docker --skip-tools
```

4. **ヘルプの表示**
```bash
./scripts/setup.sh --help
```

注意事項：
1. スクリプトを実行する前に実行権限を付与してください
   ```bash
   chmod +x scripts/setup.sh
   ```

2. スクリプトはUbuntu/Debian系のディストリビューションを想定しています

3. sudoコマンドを使用するため、必要に応じてパスワードの入力を求められます

このスクリプトにより、開発環境のセットアップを自動化し、チーム全体で一貫した環境を維持できます。