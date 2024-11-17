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
trap 'log_error "デプロイに失敗しました。エラーコード: $?"' ERR

# タイムスタンプ
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# デフォルト設定
ENVIRONMENT="development"
SKIP_TESTS=false
SKIP_BUILD=false
DEPLOY_TYPE="rolling"
SERVICE="all"
VERSION="latest"
DRY_RUN=false

# ヘルプメッセージ
show_help() {
    echo "AI-Driven Operations Platform デプロイスクリプト"
    echo
    echo "使用方法: $0 [オプション]"
    echo
    echo "オプション:"
    echo "  -h, --help                  ヘルプメッセージを表示"
    echo "  -e, --environment [ENV]     環境を指定 (development/staging/production)"
    echo "  -s, --service [SERVICE]     デプロイするサービスを指定 (all/detection/analysis/remediation)"
    echo "  -v, --version [VERSION]     バージョンを指定"
    echo "  -t, --type [TYPE]          デプロイタイプ (rolling/blue-green/canary)"
    echo "  --skip-tests               テストをスキップ"
    echo "  --skip-build              ビルドをスキップ"
    echo "  --dry-run                 実行せずに計画のみ表示"
    echo
}

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
        -s|--service)
            SERVICE="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -t|--type)
            DEPLOY_TYPE="$2"
            shift 2
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            log_error "不明なオプション: $1"
            show_help
            exit 1
            ;;
    esac
done

# 設定の検証
validate_config() {
    log_info "設定を検証中..."
    
    # 環境の検証
    case $ENVIRONMENT in
        development|staging|production)
            ;;
        *)
            log_error "無効な環境: $ENVIRONMENT"
            exit 1
            ;;
    esac
    
    # デプロイタイプの検証
    case $DEPLOY_TYPE in
        rolling|blue-green|canary)
            ;;
        *)
            log_error "無効なデプロイタイプ: $DEPLOY_TYPE"
            exit 1
            ;;
    esac
    
    # サービスの検証
    case $SERVICE in
        all|detection|analysis|remediation)
            ;;
        *)
            log_error "無効なサービス: $SERVICE"
            exit 1
            ;;
    esac
}

# テストの実行
run_tests() {
    if [ "$SKIP_TESTS" = true ]; then
        log_warning "テストをスキップします"
        return
    fi
    
    log_info "テストを実行中..."
    
    if [ "$SERVICE" = "all" ]; then
        poetry run pytest tests/
    else
        poetry run pytest tests/$SERVICE/
    fi
}

# Dockerイメージのビルド
build_images() {
    if [ "$SKIP_BUILD" = true ]; then
        log_warning "ビルドをスキップします"
        return
    fi
    
    log_info "Dockerイメージをビルド中..."
    
    if [ "$SERVICE" = "all" ]; then
        services=("detection" "analysis" "remediation")
    else
        services=("$SERVICE")
    fi
    
    for svc in "${services[@]}"; do
        log_info "Building $svc service..."
        if [ "$DRY_RUN" = false ]; then
            docker build -t "aiops/$svc:$VERSION" \
                --build-arg ENV=$ENVIRONMENT \
                --build-arg VERSION=$VERSION \
                -f $svc/Dockerfile .
        fi
    done
}

# Terraformの実行
apply_terraform() {
    log_info "Terraformを適用中..."
    
    cd infrastructure/terraform
    
    # 環境固有の変数ファイルを使用
    tfvars_file="environments/$ENVIRONMENT.tfvars"
    
    if [ ! -f "$tfvars_file" ]; then
        log_error "Terraform変数ファイルが見つかりません: $tfvars_file"
        exit 1
    fi
    
    if [ "$DRY_RUN" = false ]; then
        terraform init
        terraform plan -var-file="$tfvars_file" -out=tfplan
        terraform apply tfplan
    else
        terraform plan -var-file="$tfvars_file"
    fi
    
    cd ../../
}

# Kubernetesマニフェストの適用
apply_kubernetes() {
    log_info "Kubernetesマニフェストを適用中..."
    
    if [ "$SERVICE" = "all" ]; then
        services=("detection" "analysis" "remediation")
    else
        services=("$SERVICE")
    fi
    
    for svc in "${services[@]}"; do
        log_info "Deploying $svc service..."
        
        # 環境固有のマニフェストを使用
        manifest_dir="infrastructure/kubernetes/overlays/$ENVIRONMENT"
        
        if [ ! -d "$manifest_dir" ]; then
            log_error "Kubernetesマニフェストディレクトリが見つかりません: $manifest_dir"
            exit 1
        fi
        
        if [ "$DRY_RUN" = false ]; then
            case $DEPLOY_TYPE in
                rolling)
                    kubectl apply -k "$manifest_dir/$svc"
                    ;;
                blue-green)
                    execute_blue_green_deployment "$svc" "$manifest_dir"
                    ;;
                canary)
                    execute_canary_deployment "$svc" "$manifest_dir"
                    ;;
            esac
        else
            kubectl apply -k "$manifest_dir/$svc" --dry-run=client
        fi
    done
}

# Blue-Greenデプロイメントの実行
execute_blue_green_deployment() {
    local service=$1
    local manifest_dir=$2
    
    log_info "Blue-Greenデプロイメントを実行中: $service"
    
    # 新しいデプロイメントの作成
    kubectl apply -f "$manifest_dir/$service/deployment-green.yaml"
    
    # 新しいデプロイメントの準備待ち
    kubectl rollout status deployment/$service-green
    
    # トラフィックの切り替え
    kubectl apply -f "$manifest_dir/$service/service-green.yaml"
    
    # 古いデプロイメントの削除
    kubectl delete -f "$manifest_dir/$service/deployment-blue.yaml"
}

# Canaryデプロイメントの実行
execute_canary_deployment() {
    local service=$1
    local manifest_dir=$2
    
    log_info "Canaryデプロイメントを実行中: $service"
    
    # Canaryデプロイメントの作成
    kubectl apply -f "$manifest_dir/$service/deployment-canary.yaml"
    
    # トラフィックの段階的な移行
    for percentage in 10 30 50 80 100; do
        log_info "Canaryトラフィックを${percentage}%に設定"
        kubectl patch service/$service -p \
            "{\"spec\":{\"selector\":{\"version\":\"$VERSION\",\"weight\":\"$percentage\"}}}"
        sleep 30
    done
    
    # 古いバージョンの削除
    kubectl delete -f "$manifest_dir/$service/deployment.yaml"
}

# バックアップの作成
create_backup() {
    log_info "バックアップを作成中..."
    
    backup_dir="backups/$ENVIRONMENT/$TIMESTAMP"
    mkdir -p "$backup_dir"
    
    # 設定のバックアップ
    cp config/* "$backup_dir/"
    
    # データベースのバックアップ
    if [ "$DRY_RUN" = false ]; then
        kubectl exec -it $(kubectl get pod -l app=postgres -o jsonpath="{.items[0].metadata.name}") \
            -- pg_dump -U postgres aiops > "$backup_dir/database.sql"
    fi
}

# デプロイ後の検証
validate_deployment() {
    log_info "デプロイメントを検証中..."
    
    if [ "$SERVICE" = "all" ]; then
        services=("detection" "analysis" "remediation")
    else
        services=("$SERVICE")
    fi
    
    for svc in "${services[@]}"; do
        # ヘルスチェック
        if ! curl -s "http://localhost:8000/$svc/health" | grep -q "healthy"; then
            log_error "$svc サービスのヘルスチェックに失敗しました"
            return 1
        fi
        
        # Podのステータス確認
        if ! kubectl get pods -l app=$svc | grep -q "Running"; then
            log_error "$svc サービスのPodが実行されていません"
            return 1
        fi
    done
    
    log_info "デプロイメントの検証が完了しました"
}

# メイン処理
main() {
    log_info "デプロイを開始します: $ENVIRONMENT 環境"
    
    # 設定の検証
    validate_config
    
    # バックアップの作成
    create_backup
    
    # テストの実行
    run_tests
    
    # Dockerイメージのビルド
    build_images
    
    # インフラストラクチャの更新
    apply_terraform
    
    # Kubernetesマニフェストの適用
    apply_kubernetes
    
    # デプロイ後の検証
    if [ "$DRY_RUN" = false ]; then
        validate_deployment
    fi
    
    log_info "デプロイが完了しました!"
}

# スクリプトの実行
main

exit 0
```

このデプロイスクリプトは以下の機能を提供します：

1. **デプロイ管理**
   - 環境別のデプロイ
   - サービス別のデプロイ
   - バージョン管理
   - 段階的デプロイ

2. **デプロイ方式**
   - ローリングアップデート
   - Blue-Greenデプロイメント
   - Canaryデプロイメント

3. **検証とバックアップ**
   - 設定の検証
   - テストの実行
   - バックアップの作成
   - デプロイ後の検証

使用方法：

1. **基本的なデプロイ**
```bash
./scripts/deploy.sh -e production
```

2. **特定のサービスのデプロイ**
```bash
./scripts/deploy.sh -e staging -s detection -v 1.2.0
```

3. **Blue-Greenデプロイメント**
```bash
./scripts/deploy.sh -e production -t blue-green --skip-tests
```

4. **Dry Run**
```bash
./scripts/deploy.sh -e production --dry-run
```

注意事項：
1. スクリプトを実行する前に実行権限を付与してください
```bash
chmod +x scripts/deploy.sh
```

2. 適切なKubernetes/AWS認証情報が設定されていることを確認してください

3. 本番環境へのデプロイは慎重に行ってください

このスクリプトにより、安全で一貫性のあるデプロイプロセスを実現できます。