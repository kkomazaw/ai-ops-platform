# ベースイメージとしてPython 3.9を使用
FROM python:3.9-slim as builder

# 作業ディレクトリの設定
WORKDIR /app

# 必要なシステムパッケージのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Terraform のインストール
RUN curl -fsSL https://releases.hashicorp.com/terraform/1.3.7/terraform_1.3.7_linux_amd64.zip -o terraform.zip \
    && unzip terraform.zip \
    && mv terraform /usr/local/bin/ \
    && rm terraform.zip

# Ansible のインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    sshpass \
    openssh-client \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir ansible ansible-lint

# tflint のインストール
RUN curl -s https://raw.githubusercontent.com/terraform-linters/tflint/master/install_linux.sh | bash

# checkov のインストール
RUN pip install --no-cache-dir checkov

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
FROM python:3.9-slim as production

# 作業ディレクトリの設定
WORKDIR /app

# 必要なシステムパッケージのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-client \
    sshpass \
    && rm -rf /var/lib/apt/lists/*

# builderステージからファイルをコピー
COPY --from=builder /usr/local/bin/terraform /usr/local/bin/
COPY --from=builder /usr/local/bin/tflint /usr/local/bin/
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /app/src ./src
COPY --from=builder /app/README.md ./

# 設定ディレクトリの作成
RUN mkdir -p /app/configs /app/templates /app/output

# 環境変数の設定
ENV PYTHONPATH=/app/src \
    ANSIBLE_CONFIG=/app/configs/ansible.cfg \
    TERRAFORM_CONFIG=/app/configs/terraform.rc \
    LOG_LEVEL=INFO

# ヘルスチェック設定
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# アプリケーションの起動
CMD ["python", "-m", "src.main"]

# 開発環境用のイメージ
FROM builder as development

# 開発用パッケージのインストール
RUN poetry install --no-interaction --no-ansi

# 開発用の環境変数設定
ENV PYTHONPATH=/app/src \
    ANSIBLE_CONFIG=/app/configs/ansible.cfg \
    TERRAFORM_CONFIG=/app/configs/terraform.rc \
    LOG_LEVEL=DEBUG \
    FLASK_ENV=development

# ボリュームのマウントポイント
VOLUME ["/app/src", "/app/configs", "/app/templates", "/app/output"]

# 開発サーバーの起動
CMD ["python", "-m", "src.main", "--debug"]

# テスト環境用のイメージ
FROM builder as testing

# テスト用パッケージのインストール
RUN poetry install --no-interaction --no-ansi

# テストの実行
CMD ["pytest", "tests/", "-v", "--cov=src"]
