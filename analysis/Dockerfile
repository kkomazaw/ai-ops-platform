# ベースイメージとして最新のPython 3.9を使用
FROM python:3.9-slim AS builder

# 作業ディレクトリの設定
WORKDIR /app

# 環境変数の設定
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 必要なシステムパッケージのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Poetry のインストール
RUN curl -sSL https://install.python-poetry.org | python3 -

# 依存関係ファイルのコピー
COPY pyproject.toml poetry.lock ./

# Poetry の設定（仮想環境を作成しない）
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-dev

# ソースコードのコピー
COPY src/ ./src/
COPY README.md ./

# 本番環境用の軽量イメージ
FROM python:3.9-slim AS production

# 作業ディレクトリの設定
WORKDIR /app

# 必要なシステムパッケージのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# builderステージからファイルをコピー
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /app/src ./src
COPY --from=builder /app/README.md ./

# 必要な環境変数の設定
ENV PYTHONPATH=/app/src \
    MODEL_PATH=/app/models \
    LOG_LEVEL=INFO \
    PROMETHEUS_URL=http://prometheus:9090 \
    ELASTICSEARCH_URL=http://elasticsearch:9200

# モデルディレクトリの作成
RUN mkdir -p /app/models

# ヘルスチェック設定
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# アプリケーションの起動
CMD ["python", "-m", "src.main"]

# 開発環境用のイメージ
FROM builder AS development

# 開発用パッケージのインストール
RUN poetry install --no-interaction --no-ansi

# 開発用の環境変数設定
ENV PYTHONPATH=/app/src \
    MODEL_PATH=/app/models \
    LOG_LEVEL=DEBUG \
    FLASK_ENV=development \
    PROMETHEUS_URL=http://prometheus:9090 \
    ELASTICSEARCH_URL=http://elasticsearch:9200

# ボリュームのマウントポイント
VOLUME ["/app/src", "/app/models"]

# 開発サーバーの起動
CMD ["python", "-m", "src.main", "--debug"]

# テスト環境用のイメージ
FROM builder AS testing

# テスト用パッケージのインストール
RUN poetry install --no-interaction --no-ansi

# テストの実行
CMD ["pytest", "tests/", "-v", "--cov=src"]
