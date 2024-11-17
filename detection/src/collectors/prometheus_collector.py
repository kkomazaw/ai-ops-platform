from prometheus_api_client import PrometheusConnect
from typing import Dict, List, Optional
import logging
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrometheusCollector:
    def __init__(self, 
                 url: str,
                 basic_auth: Optional[Dict[str, str]] = None):
        """
        Prometheusメトリクス収集クラス

        Args:
            url (str): PrometheusのURL
            basic_auth (Optional[Dict[str, str]]): 基本認証情報
        """
        self.prom = PrometheusConnect(
            url=url,
            basic_auth=basic_auth
        )
        self.metric_configs = self._initialize_metric_configs()

    def _initialize_metric_configs(self) -> Dict:
        """メトリクス設定の初期化"""
        return {
            'active_users': {
                'query': 'sum(rate(http_requests_total[1m]))',
                'threshold': {'warning': 500, 'critical': 800}
            },
            'db_connections': {
                'query': 'mysql_global_status_threads_connected',
                'threshold': {'warning': 80, 'critical': 90}
            },
            'response_time': {
                'query': 'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))',
                'threshold': {'warning': 1, 'critical': 2}
            },
            'error_rate': {
                'query': 'sum(rate(http_requests_errors_total[5m])) / sum(rate(http_requests_total[5m])) * 100',
                'threshold': {'warning': 1, 'critical': 5}
            },
            'cpu_usage': {
                'query': 'sum(rate(container_cpu_usage_seconds_total[5m])) by (pod) * 100',
                'threshold': {'warning': 70, 'critical': 90}
            },
            'memory_usage': {
                'query': 'sum(container_memory_usage_bytes) by (pod) / sum(container_memory_limit_bytes) by (pod) * 100',
                'threshold': {'warning': 80, 'critical': 95}
            }
        }

    def collect_current_metrics(self) -> Dict[str, float]:
        """
        現在のメトリクスを収集

        Returns:
            Dict[str, float]: メトリクス名と値のマッピング
        """
        current_metrics = {}
        
        try:
            for metric_name, config in self.metric_configs.items():
                result = self.prom.custom_query(config['query'])
                if result:
                    # 最新の値を取得
                    value = float(result[0]['value'][1])
                    current_metrics[metric_name] = value
                    logger.debug(f"Collected {metric_name}: {value}")
                    
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            raise
            
        return current_metrics

    def collect_historical_data(self,
                              duration_hours: int = 24,
                              step_seconds: int = 60) -> Dict[str, pd.DataFrame]:
        """
        過去のメトリクスデータを収集

        Args:
            duration_hours (int): 収集する期間（時間）
            step_seconds (int): データポイント間隔（秒）

        Returns:
            Dict[str, pd.DataFrame]: メトリクス名とデータフレームのマッピング
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=duration_hours)
        historical_data = {}

        try:
            for metric_name, config in self.metric_configs.items():
                result = self.prom.custom_query_range(
                    query=config['query'],
                    start_time=start_time,
                    end_time=end_time,
                    step=step_seconds
                )
                
                if result:
                    # データフレームに変換
                    timestamps = []
                    values = []
                    for data_point in result[0]['values']:
                        timestamps.append(pd.to_datetime(data_point[0], unit='s'))
                        values.append(float(data_point[1]))
                    
                    df = pd.DataFrame({
                        'timestamp': timestamps,
                        'value': values
                    })
                    df.set_index('timestamp', inplace=True)
                    historical_data[metric_name] = df
                    
                    logger.info(f"Collected historical data for {metric_name}: {len(df)} points")
                    
        except Exception as e:
            logger.error(f"Error collecting historical data: {e}")
            raise
            
        return historical_data

    def get_metric_statistics(self, metric_name: str, duration_hours: int = 24) -> Dict:
        """
        メトリクスの統計情報を計算

        Args:
            metric_name (str): メトリクス名
            duration_hours (int): 計算対象期間（時間）

        Returns:
            Dict: 統計情報
        """
        try:
            historical_data = self.collect_historical_data(
                duration_hours=duration_hours,
                step_seconds=300  # 5分間隔
            )
            
            if metric_name in historical_data:
                df = historical_data[metric_name]
                values = df['value'].values
                
                return {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'percentile_95': float(np.percentile(values, 95)),
                    'count': len(values),
                    'threshold': self.metric_configs[metric_name]['threshold']
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error calculating statistics for {metric_name}: {e}")
            raise

    def check_metric_health(self, metric_name: str, value: float) -> Dict:
        """
        メトリクスの健全性チェック

        Args:
            metric_name (str): メトリクス名
            value (float): チェック対象の値

        Returns:
            Dict: 健全性評価結果
        """
        if metric_name not in self.metric_configs:
            raise ValueError(f"Unknown metric: {metric_name}")
            
        threshold = self.metric_configs[metric_name]['threshold']
        
        status = 'healthy'
        if value >= threshold['critical']:
            status = 'critical'
        elif value >= threshold['warning']:
            status = 'warning'
            
        return {
            'metric': metric_name,
            'value': value,
            'status': status,
            'threshold': threshold
        }

    def get_available_metrics(self) -> List[str]:
        """
        利用可能なメトリクス一覧を取得

        Returns:
            List[str]: メトリクス名のリスト
        """
        return list(self.metric_configs.keys())

    def add_metric_config(self, 
                         metric_name: str,
                         query: str,
                         warning_threshold: float,
                         critical_threshold: float) -> None:
        """
        新しいメトリクス設定を追加

        Args:
            metric_name (str): メトリクス名
            query (str): Prometheusクエリ
            warning_threshold (float): 警告閾値
            critical_threshold (float): 危険閾値
        """
        self.metric_configs[metric_name] = {
            'query': query,
            'threshold': {
                'warning': warning_threshold,
                'critical': critical_threshold
            }
        }
        logger.info(f"Added new metric configuration: {metric_name}")

    def update_threshold(self,
                        metric_name: str,
                        warning_threshold: Optional[float] = None,
                        critical_threshold: Optional[float] = None) -> None:
        """
        メトリクスの閾値を更新

        Args:
            metric_name (str): メトリクス名
            warning_threshold (Optional[float]): 新しい警告閾値
            critical_threshold (Optional[float]): 新しい危険閾値
        """
        if metric_name not in self.metric_configs:
            raise ValueError(f"Unknown metric: {metric_name}")
            
        if warning_threshold is not None:
            self.metric_configs[metric_name]['threshold']['warning'] = warning_threshold
            
        if critical_threshold is not None:
            self.metric_configs[metric_name]['threshold']['critical'] = critical_threshold
            
        logger.info(f"Updated thresholds for {metric_name}")
