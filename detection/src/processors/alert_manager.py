from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import requests
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading
from queue import Queue
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """アラートの重要度"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertStatus(Enum):
    """アラートのステータス"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SILENCED = "silenced"

@dataclass
class AlertConfig:
    """アラート設定"""
    name: str
    description: str
    severity: AlertSeverity
    notification_channels: List[str]
    cooldown_minutes: int
    grouping_key: Optional[str] = None
    auto_resolve_minutes: Optional[int] = None

@dataclass
class Alert:
    """アラート情報"""
    id: str
    config_name: str
    severity: AlertSeverity
    status: AlertStatus
    summary: str
    details: Dict
    created_at: datetime
    updated_at: datetime
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    notification_sent: bool = False

class AlertManager:
    def __init__(self, config_path: Optional[str] = None):
        """
        アラートマネージャーの初期化

        Args:
            config_path (Optional[str]): 設定ファイルのパス
        """
        self.alert_configs = self._load_alert_configs(config_path)
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_queue = Queue()
        self.last_notification_times: Dict[str, datetime] = {}
        
        # 通知処理用のワーカースレッドを開始
        self.notification_thread = threading.Thread(
            target=self._notification_worker,
            daemon=True
        )
        self.notification_thread.start()

    def _load_alert_configs(self, config_path: Optional[str]) -> Dict[str, AlertConfig]:
        """アラート設定の読み込み"""
        if config_path:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        else:
            # デフォルト設定
            config_data = {
                "high_cpu_usage": {
                    "name": "high_cpu_usage",
                    "description": "CPU usage exceeds threshold",
                    "severity": "critical",
                    "notification_channels": ["slack", "email"],
                    "cooldown_minutes": 15,
                    "auto_resolve_minutes": 60
                },
                "memory_warning": {
                    "name": "memory_warning",
                    "description": "Memory usage is high",
                    "severity": "warning",
                    "notification_channels": ["slack"],
                    "cooldown_minutes": 30
                },
                "response_time_critical": {
                    "name": "response_time_critical",
                    "description": "Response time is critically high",
                    "severity": "emergency",
                    "notification_channels": ["slack", "email", "pagerduty"],
                    "cooldown_minutes": 5,
                    "auto_resolve_minutes": 30
                }
            }

        configs = {}
        for name, cfg in config_data.items():
            configs[name] = AlertConfig(
                name=cfg["name"],
                description=cfg["description"],
                severity=AlertSeverity(cfg["severity"]),
                notification_channels=cfg["notification_channels"],
                cooldown_minutes=cfg["cooldown_minutes"],
                auto_resolve_minutes=cfg.get("auto_resolve_minutes")
            )
        return configs

    def create_alert(self, 
                    config_name: str,
                    summary: str,
                    details: Dict) -> Optional[Alert]:
        """
        アラートの作成

        Args:
            config_name (str): アラート設定名
            summary (str): アラートの概要
            details (Dict): アラートの詳細情報

        Returns:
            Optional[Alert]: 作成されたアラート
        """
        try:
            if config_name not in self.alert_configs:
                logger.error(f"Unknown alert config: {config_name}")
                return None

            config = self.alert_configs[config_name]
            
            # クールダウン時間のチェック
            if self._is_in_cooldown(config_name):
                logger.info(f"Alert {config_name} is in cooldown period")
                return None

            # アラートIDの生成
            alert_id = f"{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            alert = Alert(
                id=alert_id,
                config_name=config_name,
                severity=config.severity,
                status=AlertStatus.ACTIVE,
                summary=summary,
                details=details,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )

            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # 通知キューに追加
            self.notification_queue.put(alert)
            
            logger.info(f"Created alert: {alert_id}")
            return alert

        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            return None

    def _is_in_cooldown(self, config_name: str) -> bool:
        """クールダウン期間中かどうかのチェック"""
        if config_name not in self.last_notification_times:
            return False

        config = self.alert_configs[config_name]
        cooldown_delta = timedelta(minutes=config.cooldown_minutes)
        return (datetime.now() - self.last_notification_times[config_name]) < cooldown_delta

    def _notification_worker(self):
        """通知処理ワーカー"""
        while True:
            try:
                alert = self.notification_queue.get()
                self._send_notifications(alert)
                self.notification_queue.task_done()
            except Exception as e:
                logger.error(f"Error in notification worker: {e}")
            time.sleep(1)

    def _send_notifications(self, alert: Alert):
        """通知の送信"""
        config = self.alert_configs[alert.config_name]
        
        for channel in config.notification_channels:
            try:
                if channel == "slack":
                    self._send_slack_notification(alert)
                elif channel == "email":
                    self._send_email_notification(alert)
                elif channel == "pagerduty":
                    self._send_pagerduty_notification(alert)
            except Exception as e:
                logger.error(f"Error sending {channel} notification: {e}")

        self.last_notification_times[alert.config_name] = datetime.now()
        alert.notification_sent = True

    def _send_slack_notification(self, alert: Alert):
        """Slack通知の送信"""
        try:
            webhook_url = "your_slack_webhook_url"  # 環境変数から取得することを推奨
            
            message = {
                "attachments": [{
                    "color": self._get_severity_color(alert.severity),
                    "title": f"Alert: {alert.summary}",
                    "text": json.dumps(alert.details, indent=2),
                    "fields": [
                        {
                            "title": "Severity",
                            "value": alert.severity.value,
                            "short": True
                        },
                        {
                            "title": "Status",
                            "value": alert.status.value,
                            "short": True
                        }
                    ]
                }]
            }
            
            response = requests.post(webhook_url, json=message)
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            raise

    def _send_email_notification(self, alert: Alert):
        """メール通知の送信"""
        try:
            smtp_server = "smtp.your-server.com"  # 環境変数から取得することを推奨
            smtp_port = 587
            smtp_user = "your-email@example.com"
            smtp_password = "your-password"
            
            msg = MIMEMultipart()
            msg['Subject'] = f"Alert: {alert.summary}"
            msg['From'] = smtp_user
            msg['To'] = "target@example.com"
            
            body = f"""
Alert Details:
- Severity: {alert.severity.value}
- Status: {alert.status.value}
- Created At: {alert.created_at}

Details:
{json.dumps(alert.details, indent=2)}
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)
                
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            raise

    def _send_pagerduty_notification(self, alert: Alert):
        """PagerDuty通知の送信"""
        try:
            api_key = "your-pagerduty-api-key"  # 環境変数から取得することを推奨
            
            payload = {
                "incident": {
                    "type": "incident",
                    "title": alert.summary,
                    "service": {
                        "id": "your-service-id",
                        "type": "service_reference"
                    },
                    "urgency": "high" if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY] else "low",
                    "body": {
                        "type": "incident_body",
                        "details": json.dumps(alert.details, indent=2)
                    }
                }
            }
            
            headers = {
                "Accept": "application/vnd.pagerduty+json;version=2",
                "Authorization": f"Token token={api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                "https://api.pagerduty.com/incidents",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Error sending PagerDuty notification: {e}")
            raise

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """アラートの確認"""
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_by = acknowledged_by
        alert.updated_at = datetime.now()
        
        logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
        return True

    def resolve_alert(self, alert_id: str) -> bool:
        """アラートの解決"""
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()
        alert.updated_at = datetime.now()
        
        del self.active_alerts[alert_id]
        logger.info(f"Alert {alert_id} resolved")
        return True

    def get_active_alerts(self, 
                         severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """アクティブなアラートの取得"""
        alerts = list(self.active_alerts.values())
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return sorted(alerts, key=lambda x: x.created_at, reverse=True)

    def get_alert_history(self,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> List[Alert]:
        """アラート履歴の取得"""
        alerts = self.alert_history
        
        if start_time:
            alerts = [a for a in alerts if a.created_at >= start_time]
        if end_time:
            alerts = [a for a in alerts if a.created_at <= end_time]
            
        return sorted(alerts, key=lambda x: x.created_at, reverse=True)

    def _get_severity_color(self, severity: AlertSeverity) -> str:
        """重要度に応じた色コードの取得"""
        color_map = {
            AlertSeverity.INFO: "#3498db",
            AlertSeverity.WARNING: "#f1c40f",
            AlertSeverity.CRITICAL: "#e74c3c",
            AlertSeverity.EMERGENCY: "#c0392b"
        }
        return color_map.get(severity, "#95a5a6")

    def cleanup_old_alerts(self, days: int = 30):
        """古いアラート履歴のクリーンアップ"""
        cutoff_date = datetime.now() - timedelta(days=days)
        self.alert_history = [
            alert for alert in self.alert_history
            if alert.created_at >= cutoff_date
        ]
        logger.info(f"Cleaned up alerts older than {days} days")

    def get_alert_stats(self) -> Dict:
        """アラート統計の取得"""
        total_alerts = len(self.alert_history)
        active_alerts = len(self.active_alerts)
        
        severity_counts = {
            severity: len([a for a in self.alert_history if a.severity == severity])
            for severity in AlertSeverity
        }
        
        return {
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "severity_distribution": severity_counts,
            "recent_alerts": len([
                a for a in self.alert_history
                if a.created_at >= datetime.now() - timedelta(hours=24)
            ])
        }
