import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import logging
from dataclasses import dataclass
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ThresholdConfig:
    """閾値設定"""
    warning: float
    critical: float
    method: str = "static"  # static or dynamic

class StatisticalDetector:
    def __init__(self, 
                 window_size: int = 60,
                 threshold_std: float = 3):
        """
        統計的異常検知モデル

        Args:
            window_size (int): 移動窓のサイズ
            threshold_std (float): 異常判定の標準偏差閾値
        """
        self.window_size = window_size
        self.threshold_std = threshold_std
        self.metric_windows = {}
        self.thresholds = {}

    def configure_metric(self, 
                        metric_name: str,
                        threshold_config: ThresholdConfig) -> None:
        """
        メトリクスごとの設定

        Args:
            metric_name (str): メトリクス名
            threshold_config (ThresholdConfig): 閾値設定
        """
        self.metric_windows[metric_name] = deque(maxlen=self.window_size)
        self.thresholds[metric_name] = threshold_config

    def update_metrics(self, metrics: Dict[str, float]) -> List[Dict]:
        """
        メトリクスの更新と異常検知

        Args:
            metrics (Dict[str, float]): 新しいメトリクス値

        Returns:
            List[Dict]: 検知された異常のリスト
        """
        anomalies = []
        
        for metric_name, value in metrics.items():
            if metric_name in self.metric_windows:
                self.metric_windows[metric_name].append(value)
                anomaly = self._check_anomaly(metric_name, value)
                if anomaly:
                    anomalies.append(anomaly)

        return anomalies

    def _check_anomaly(self, 
                      metric_name: str,
                      value: float) -> Dict:
        """
        個別メトリクスの異常チェック

        Args:
            metric_name (str): メトリクス名
            value (float): チェック対象の値

        Returns:
            Dict: 異常検知結果
        """
        window = list(self.metric_windows[metric_name])
        
        if len(window) < 2:
            return None

        # 統計値の計算
        mean = np.mean(window[:-1])  # 最新値を除く
        std = np.std(window[:-1])
        z_score = (value - mean) / std if std > 0 else 0
        
        # 閾値チェック
        threshold_config = self.thresholds[metric_name]
        
        if threshold_config.method == "dynamic":
            is_anomaly = abs(z_score) > self.threshold_std
            severity = self._calculate_severity(z_score)
        else:
            is_anomaly = value > threshold_config.warning
            severity = (
                'CRITICAL' if value > threshold_config.critical
                else 'WARNING' if value > threshold_config.warning
                else 'NORMAL'
            )

        if is_anomaly:
            return {
                'metric_name': metric_name,
                'value': value,
                'mean': mean,
                'std': std,
                'z_score': z_score,
                'severity': severity,
                'threshold_type': threshold_config.method
            }
        
        return None

    def _calculate_severity(self, z_score: float) -> str:
        """
        Z-scoreに基づく重要度の計算

        Args:
            z_score (float): Z-score値

        Returns:
            str: 重要度レベル
        """
        abs_score = abs(z_score)
        if abs_score > 5:
            return 'CRITICAL'
        elif abs_score > 4:
            return 'HIGH'
        elif abs_score > 3:
            return 'MEDIUM'
        else:
            return 'LOW'

    def calculate_baseline(self, 
                         historical_data: Dict[str, List[float]]) -> None:
        """
        過去データからベースラインを計算

        Args:
            historical_data (Dict[str, List[float]]): 過去のメトリクスデータ
        """
        for metric_name, values in historical_data.items():
            if metric_name in self.thresholds:
                config = self.thresholds[metric_name]
                if config.method == "dynamic":
                    mean = np.mean(values)
                    std = np.std(values)
                    config.warning = mean + (2 * std)
                    config.critical = mean + (3 * std)
                    logger.info(
                        f"Updated {metric_name} thresholds - "
                        f"Warning: {config.warning:.2f}, "
                        f"Critical: {config.critical:.2f}"
                    )

    def get_current_stats(self) -> Dict[str, Dict]:
        """
        現在の統計情報を取得

        Returns:
            Dict[str, Dict]: メトリクスごとの統計情報
        """
        stats = {}
        for metric_name, window in self.metric_windows.items():
            if len(window) > 0:
                stats[metric_name] = {
                    'current_value': window[-1],
                    'mean': np.mean(window),
                    'std': np.std(window),
                    'min': np.min(window),
                    'max': np.max(window),
                    'threshold_config': {
                        'warning': self.thresholds[metric_name].warning,
                        'critical': self.thresholds[metric_name].critical,
                        'method': self.thresholds[metric_name].method
                    }
                }
        return stats
