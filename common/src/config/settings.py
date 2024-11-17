from typing import Dict, Any, Optional
from pathlib import Path
import os
import yaml
import json
from pydantic import BaseSettings, Field
from functools import lru_cache
import logging
from dotenv import load_dotenv
import socket
from dataclasses import dataclass

# 環境変数の読み込み
load_dotenv()

# ロガーの設定
logger = logging.getLogger(__name__)

class LogConfig(BaseSettings):
    """ログ設定"""
    LEVEL: str = Field(default="INFO")
    FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    FILE_PATH: Optional[str] = Field(default=None)
    ROTATION: str = Field(default="10 MB")
    RETENTION: str = Field(default="30 days")

class DatabaseConfig(BaseSettings):
    """データベース設定"""
    HOST: str = Field(default="localhost")
    PORT: int = Field(default=5432)
    USERNAME: str = Field(...)
    PASSWORD: str = Field(...)
    DATABASE: str = Field(...)
    POOL_SIZE: int = Field(default=5)
    MAX_OVERFLOW: int = Field(default=10)
    POOL_TIMEOUT: int = Field(default=30)

class RedisConfig(BaseSettings):
    """Redis設定"""
    HOST: str = Field(default="localhost")
    PORT: int = Field(default=6379)
    PASSWORD: Optional[str] = Field(default=None)
    DB: int = Field(default=0)
    SSL: bool = Field(default=False)

class PrometheusConfig(BaseSettings):
    """Prometheus設定"""
    ENABLED: bool = Field(default=True)
    PORT: int = Field(default=9090)
    PUSH_GATEWAY: Optional[str] = Field(default=None)
    EXPORT_INTERVAL: int = Field(default=15)

class ElasticsearchConfig(BaseSettings):
    """Elasticsearch設定"""
    HOSTS: list = Field(default=["http://localhost:9200"])
    USERNAME: Optional[str] = Field(default=None)
    PASSWORD: Optional[str] = Field(default=None)
    INDEX_PREFIX: str = Field(default="app-logs-")
    SHARDS: int = Field(default=1)
    REPLICAS: int = Field(default=1)

class AWSConfig(BaseSettings):
    """AWS設定"""
    REGION: str = Field(default="us-west-2")
    ACCESS_KEY_ID: Optional[str] = Field(default=None)
    SECRET_ACCESS_KEY: Optional[str] = Field(default=None)
    ROLE_ARN: Optional[str] = Field(default=None)
    S3_BUCKET: Optional[str] = Field(default=None)

class Settings(BaseSettings):
    """アプリケーション設定"""
    # 基本設定
    APP_NAME: str = Field(default="ai-ops-platform")
    ENV: str = Field(default="development")
    DEBUG: bool = Field(default=False)
    SECRET_KEY: str = Field(...)
    API_VERSION: str = Field(default="v1")
    
    # サーバー設定
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    WORKERS: int = Field(default=4)
    TIMEOUT: int = Field(default=60)
    
    # コンポーネント設定
    LOG: LogConfig = LogConfig()
    DATABASE: DatabaseConfig = DatabaseConfig()
    REDIS: RedisConfig = RedisConfig()
    PROMETHEUS: PrometheusConfig = PrometheusConfig()
    ELASTICSEARCH: ElasticsearchConfig = ElasticsearchConfig()
    AWS: AWSConfig = AWSConfig()
    
    # セキュリティ設定
    ALLOWED_HOSTS: list = Field(default=["*"])
    CORS_ORIGINS: list = Field(default=["*"])
    JWT_SECRET: str = Field(...)
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_EXPIRATION: int = Field(default=3600)  # 1時間
    
    # キャッシュ設定
    CACHE_TTL: int = Field(default=300)  # 5分
    CACHE_PREFIX: str = Field(default="aiops:")
    
    # 監視設定
    HEALTH_CHECK_INTERVAL: int = Field(default=60)  # 1分
    METRIC_COLLECTION_INTERVAL: int = Field(default=15)  # 15秒
    
    # 自動修復設定
    AUTO_REMEDIATION_ENABLED: bool = Field(default=False)
    MAX_AUTO_REMEDIATION_ATTEMPTS: int = Field(default=3)
    REMEDIATION_TIMEOUT: int = Field(default=300)  # 5分
    
    class Config:
        env_file = '.env'
        case_sensitive = True

class ConfigurationManager:
    """設定管理クラス"""
    _instance = None
    
    def __new__(cls):
        """シングルトンパターンの実装"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """設定管理の初期化"""
        self.settings = self._load_settings()
        self.environment = os.getenv('ENVIRONMENT', 'development')
        self.config_path = Path('config')
        self.custom_settings = {}

    @lru_cache()
    def _load_settings(self) -> Settings:
        """設定のロード"""
        try:
            return Settings()
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            raise

    def load_yaml_config(self, filename: str) -> Dict:
        """YAML設定ファイルのロード"""
        try:
            config_file = self.config_path / f"{filename}.yaml"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading YAML config {filename}: {e}")
            return {}

    def update_settings(self, updates: Dict[str, Any]) -> None:
        """設定の更新"""
        try:
            for key, value in updates.items():
                if hasattr(self.settings, key):
                    setattr(self.settings, key, value)
                else:
                    self.custom_settings[key] = value
            logger.info("Settings updated successfully")
        except Exception as e:
            logger.error(f"Error updating settings: {e}")
            raise

    def get_service_config(self, service_name: str) -> Dict:
        """サービス固有の設定を取得"""
        try:
            service_config = self.load_yaml_config(f"services/{service_name}")
            env_config = self.load_yaml_config(f"environments/{self.environment}")
            
            # 環境設定とサービス設定のマージ
            merged_config = {**service_config, **env_config.get(service_name, {})}
            return merged_config
            
        except Exception as e:
            logger.error(f"Error loading service config for {service_name}: {e}")
            return {}

    def get_database_url(self) -> str:
        """データベースURLの生成"""
        db = self.settings.DATABASE
        return (
            f"postgresql://{db.USERNAME}:{db.PASSWORD}@"
            f"{db.HOST}:{db.PORT}/{db.DATABASE}"
        )

    def get_redis_url(self) -> str:
        """RedisURLの生成"""
        redis = self.settings.REDIS
        auth = f":{redis.PASSWORD}@" if redis.PASSWORD else "@"
        return f"redis{'s' if redis.SSL else ''}://{auth}{redis.HOST}:{redis.PORT}/{redis.DB}"

    def export_settings(self, format: str = 'json') -> str:
        """設定のエクスポート"""
        try:
            settings_dict = {
                **self.settings.dict(),
                **self.custom_settings
            }
            
            if format == 'json':
                return json.dumps(settings_dict, indent=2)
            elif format == 'yaml':
                return yaml.dump(settings_dict, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting settings: {e}")
            raise

    @property
    def hostname(self) -> str:
        """ホスト名の取得"""
        return socket.gethostname()

    @property
    def is_production(self) -> bool:
        """本番環境かどうかの判定"""
        return self.environment == 'production'

    @property
    def debug_mode(self) -> bool:
        """デバッグモードの判定"""
        return self.settings.DEBUG and not self.is_production

# グローバル設定マネージャーのインスタンス
config = ConfigurationManager()

# 使用例
if __name__ == "__main__":
    # 設定の取得
    print(f"Application Name: {config.settings.APP_NAME}")
    print(f"Environment: {config.environment}")
    print(f"Debug Mode: {config.debug_mode}")
    
    # データベースURL
    print(f"Database URL: {config.get_database_url()}")
    
    # サービス設定
    detection_config = config.get_service_config("detection")
    print(f"Detection Service Config: {detection_config}")
    
    # 設定のエクスポート
    settings_json = config.export_settings(format='json')
    print(f"Settings JSON: {settings_json}")
```

このSettings/ConfigurationManagerモジュールは以下の機能を提供します：

1. **設定管理**
   - 環境変数
   - 設定ファイル
   - デフォルト値
   - 型安全性

2. **コンポーネント設定**
   - データベース
   - Redis
   - Prometheus
   - Elasticsearch
   - AWS

3. **環境管理**
   - 開発/本番環境
   - デバッグモード
   - サービス固有設定

使用例：
```python
from common.src.config.settings import config

# 基本設定の取得
app_name = config.settings.APP_NAME
debug_mode = config.debug_mode

# データベース接続情報
db_url = config.get_database_url()

# サービス固有の設定
service_config = config.get_service_config("my_service")

# カスタム設定の更新
config.update_settings({
    "CUSTOM_SETTING": "value"
})

# 設定のエクスポート
settings_yaml = config.export_settings(format='yaml')
