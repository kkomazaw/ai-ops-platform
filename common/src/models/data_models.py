from typing import List, Dict, Optional, Union, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid
import json

class Severity(str, Enum):
    """重要度レベル"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class Status(str, Enum):
    """ステータス"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"

class ResourceType(str, Enum):
    """リソースタイプ"""
    EC2_INSTANCE = "ec2_instance"
    RDS_INSTANCE = "rds_instance"
    S3_BUCKET = "s3_bucket"
    LAMBDA_FUNCTION = "lambda_function"
    DYNAMODB_TABLE = "dynamodb_table"
    ECS_SERVICE = "ecs_service"
    EKS_CLUSTER = "eks_cluster"

class MetricData(BaseModel):
    """メトリクスデータモデル"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = Field(default_factory=dict)
    unit: Optional[str] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class IncidentEvent(BaseModel):
    """インシデントイベントモデル"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    severity: Severity
    title: str
    description: str
    source: str
    affected_resources: List[str] = Field(default_factory=list)
    metrics: List[MetricData] = Field(default_factory=list)
    tags: Dict[str, str] = Field(default_factory=dict)

    @validator('severity')
    def validate_severity(cls, v):
        if isinstance(v, str):
            return Severity(v.lower())
        return v

class RemediationAction(BaseModel):
    """修復アクションモデル"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    incident_id: str
    action_type: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    status: Status = Status.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    executed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    def update_status(self, status: Status):
        """ステータスの更新"""
        self.status = status
        self.updated_at = datetime.utcnow()
        if status == Status.COMPLETED:
            self.completed_at = datetime.utcnow()

class ResourceConfig(BaseModel):
    """リソース設定モデル"""
    resource_id: str
    resource_type: ResourceType
    region: str
    configuration: Dict[str, Any]
    tags: Dict[str, str] = Field(default_factory=dict)
    last_modified: datetime = Field(default_factory=datetime.utcnow)

    @validator('resource_type')
    def validate_resource_type(cls, v):
        if isinstance(v, str):
            return ResourceType(v.lower())
        return v

class AuditLog(BaseModel):
    """監査ログモデル"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    action: str
    resource_id: str
    resource_type: str
    user_id: str
    changes: Dict[str, Any] = Field(default_factory=dict)
    status: Status
    details: Optional[str] = None

class AnalysisResult(BaseModel):
    """分析結果モデル"""
    incident_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    root_cause: str
    confidence_score: float
    affected_components: List[str]
    evidence: Dict[str, Any]
    recommended_actions: List[Dict[str, Any]]
    related_incidents: List[str] = Field(default_factory=list)

class MonitoringRule(BaseModel):
    """監視ルールモデル"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    resource_type: ResourceType
    metrics: List[str]
    conditions: Dict[str, Any]
    severity: Severity
    actions: List[Dict[str, Any]]
    enabled: bool = True
    last_updated: datetime = Field(default_factory=datetime.utcnow)

class NotificationConfig(BaseModel):
    """通知設定モデル"""
    channel_type: str  # slack, email, webhook等
    channel_config: Dict[str, Any]
    notification_rules: Dict[Severity, bool]
    templates: Dict[str, str]
    enabled: bool = True

class ServiceDependency(BaseModel):
    """サービス依存関係モデル"""
    source_service: str
    target_service: str
    dependency_type: str  # sync, async等
    criticality: str  # high, medium, low
    health_check_endpoint: Optional[str] = None
    timeout_seconds: Optional[int] = None

class HealthStatus(BaseModel):
    """ヘルスステータスモデル"""
    service_name: str
    status: Status
    last_check: datetime = Field(default_factory=datetime.utcnow)
    metrics: Dict[str, float]
    dependencies: List[ServiceDependency] = Field(default_factory=list)
    incidents: List[str] = Field(default_factory=list)

class MaintenanceWindow(BaseModel):
    """メンテナンスウィンドウモデル"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime
    end_time: datetime
    services: List[str]
    description: str
    status: Status = Status.PENDING
    notifications_sent: bool = False

class BackupConfig(BaseModel):
    """バックアップ設定モデル"""
    resource_id: str
    resource_type: ResourceType
    schedule: str  # cron形式
    retention_days: int
    backup_type: str  # full, incremental等
    storage_location: str
    last_backup: Optional[datetime] = None
    next_backup: Optional[datetime] = None

def model_to_dict(model: BaseModel) -> Dict:
    """モデルを辞書に変換"""
    return json.loads(model.json())

def dict_to_model(data: Dict, model_class: type) -> BaseModel:
    """辞書からモデルを生成"""
    return model_class(**data)

# 使用例
if __name__ == "__main__":
    # インシデントイベントの作成
    incident = IncidentEvent(
        severity=Severity.ERROR,
        title="Database Connection Error",
        description="Multiple connection timeouts detected",
        source="rds-monitoring",
        affected_resources=["db-instance-1"],
        metrics=[
            MetricData(
                name="connection_errors",
                value=15.0,
                timestamp=datetime.utcnow(),
                labels={"database": "prod"}
            )
        ],
        tags={"environment": "production"}
    )

    # 修復アクションの作成
    action = RemediationAction(
        incident_id=incident.id,
        action_type="restart_instance",
        description="Restart the database instance",
        parameters={"instance_id": "db-instance-1"}
    )

    # モデルの検証と変換
    incident_dict = model_to_dict(incident)
    recovered_incident = dict_to_model(incident_dict, IncidentEvent)

    print(f"Created incident: {incident.id}")
    print(f"With action: {action.id}")
    print(f"Incident severity: {incident.severity}")

