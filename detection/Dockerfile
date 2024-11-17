# ベースイメージとしてPython 3.9を使用
FROM python:3.9-slim as builder

# 作業ディレクトリの設定
WORKDIR /app

# 必要なパッケージのインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Poetry のインストール
RUN pip install --no-cache-dir poetry==1.4.2

# Poetry の設定（仮想環境を作成しない）
RUN poetry config virtualenvs.create false

# 依存関係ファイルのコピー
COPY pyproject.toml poetry.lock ./

# 依存関係のインストール（開発用パッケージを除く）
RUN poetry install --no-dev --no-root

# ソースコードのコピー
COPY src/ ./src/
COPY README.md ./

# パッケージのインストール
RUN poetry install --no-dev

# 本番環境用の軽量イメージ
FROM python:3.9-slim as production

# 作業ディレクトリの設定
WORKDIR /app

# 必要な実行時パッケージのインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# builderステージからファイルをコピー
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /app/src ./src
COPY --from=builder /app/README.md ./

# 環境変数の設定
ENV PYTHONPATH=/app/src
ENV PROMETHEUS_URL=http://prometheus:9090
ENV ELASTICSEARCH_URL=http://elasticsearch:9200
ENV LOG_LEVEL=INFO

# ヘルスチェック用の設定
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# アプリケーションの起動
CMD ["python", "-m", "src.main"]

# 開発環境用のステージ
FROM builder as development

# 開発用パッケージのインストール
RUN poetry install --with dev

# 開発用の環境変数設定
ENV PYTHONPATH=/app/src
ENV PROMETHEUS_URL=http://prometheus:9090
ENV ELASTICSEARCH_URL=http://elasticsearch:9200
ENV LOG_LEVEL=DEBUG
ENV FLASK_ENV=development

# ボリュームのマウントポイント
VOLUME ["/app/src"]

# 開発サーバーの起動
CMD ["python", "-m", "src.main", "--debug"]
