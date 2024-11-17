import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy import stats
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MetricConfig:
    """メトリクス設定"""
    name: str
    aggregation_method: str  # avg, sum, max, min
    window_size: int  # seconds
    smoothing_factor: float  # 0-1
    threshold_warning: float
    threshold_critical: float

class MetricProcessor:
    def __init__(self):
        """メトリクス処理クラスの初期化"""
        self.metric_configs = self._initialize_metric_configs()
        self.metric_buffers = self._initialize_metric_buffers()
        self.baseline_stats = {}
        self.trend_detectors = {}

    def _initialize_metric_configs(self) -> Dict[str, MetricConfig]:
        """メトリクス設定の初期化"""
        return {
            'active_users': MetricConfig(
                name='active_users',
                aggregation_method='avg',
                window_size=300,  # 5分
                smoothing_factor=0.3,
                threshold_warning=500,
                threshold_critical=800
            ),
            'response_time': MetricConfig(
                name='response_time',
                aggregation_method='max',
                window_size=60,  # 1分
                smoothing_factor=0.2,
                threshold_warning=1000,
                threshold_critical=2000
            ),
            'error_rate': MetricConfig(
                name='error_rate',
                aggregation_method='avg',
                window_size=300,  # 5分
                smoothing_factor=0.1,
                threshold_warning=1.0,
                threshold_critical=5.0
            ),
            'cpu_usage': MetricConfig(
                name='cpu_usage',
                aggregation_method='avg',
                window_size=300,  # 5分
                smoothing_factor=0.2,
                threshold_warning=70,
                threshold_critical=90
            ),
            'memory_usage': MetricConfig(
                name='memory_usage',
                aggregation_method='avg',
                window_size=300,  # 5分
                smoothing_factor=0.2,
                threshold_warning=80,
                threshold_critical=95
            ),
            'db_connections': MetricConfig(
                name='db_connections',
                aggregation_method='max',
                window_size=60,  # 1分
                smoothing_factor=0.1,
                threshold_warning=80,
                threshold_critical=90
            )
        }

    def _initialize_metric_buffers(self) -> Dict[str, deque]:
        """メトリクスバッファの初期化"""
        buffers = {}
        for metric_name, config in self.metric_configs.items():
            # 設定された時間枠に基づいてバッファサイズを計算
            buffer_size = max(
                int(config.window_size / 10),  # 10秒ごとのサンプリングを想定
                100  # 最小バッファサイズ
            )
            buffers[metric_name] = deque(maxlen=buffer_size)
        return buffers

    def process_metrics(self, metrics: Dict[str, float]) -> Dict[str, Dict]:
        """
        メトリクスの処理

        Args:
            metrics (Dict[str, float]): 生メトリクスデータ

        Returns:
            Dict[str, Dict]: 処理済みメトリクス
        """
        processed_metrics = {}
        
        try:
            for metric_name, value in metrics.items():
                if metric_name in self.metric_configs:
                    # メトリクスバッファの更新
                    self.metric_buffers[metric_name].append(value)
                    
                    # 統計値の計算
                    stats = self._calculate_statistics(metric_name)
                    
                    # 異常値の検出
                    anomaly_score = self._detect_anomalies(metric_name, value, stats)
                    
                    # トレンド分析
                    trend = self._analyze_trend(metric_name)
                    
                    processed_metrics[metric_name] = {
                        'current_value': value,
                        'smoothed_value': self._calculate_smoothed_value(metric_name, value),
                        'statistics': stats,
                        'anomaly_score': anomaly_score,
                        'trend': trend,
                        'status': self._determine_status(metric_name, value)
                    }
                    
        except Exception as e:
            logger.error(f"Error processing metrics: {e}")
            raise
            
        return processed_metrics

    def _calculate_statistics(self, metric_name: str) -> Dict:
        """統計値の計算"""
        buffer = list(self.metric_buffers[metric_name])
        if not buffer:
            return {}

        return {
            'mean': float(np.mean(buffer)),
            'std': float(np.std(buffer)),
            'min': float(np.min(buffer)),
            'max': float(np.max(buffer)),
            'median': float(np.median(buffer)),
            'percentile_95': float(np.percentile(buffer, 95)),
            'count': len(buffer)
        }

    def _detect_anomalies(self, 
                         metric_name: str,
                         value: float,
                         stats: Dict) -> float:
        """異常値の検出"""
        if not stats or 'std' not in stats or stats['std'] == 0:
            return 0.0

        z_score = (value - stats['mean']) / stats['std']
        config = self.metric_configs[metric_name]
        
        # 異常スコアの計算（0-1の範囲に正規化）
        anomaly_score = min(abs(z_score) / 3.0, 1.0)
        
        return float(anomaly_score)

    def _analyze_trend(self, metric_name: str) -> Dict:
        """トレンド分析"""
        buffer = list(self.metric_buffers[metric_name])
        if len(buffer) < 2:
            return {'direction': 'stable', 'strength': 0.0}

        # 傾きの計算
        x = np.arange(len(buffer))
        slope, _, r_value, _, _ = stats.linregress(x, buffer)
        
        # トレンドの方向と強さの判定
        direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
        strength = abs(r_value)  # 相関係数を強さとして使用

        return {
            'direction': direction,
            'strength': float(strength),
            'slope': float(slope)
        }

    def _calculate_smoothed_value(self,
                                metric_name: str,
                                value: float) -> float:
        """指数移動平均によるスムージング"""
        config = self.metric_configs[metric_name]
        buffer = list(self.metric_buffers[metric_name])
        
        if not buffer:
            return value

        alpha = config.smoothing_factor
        prev_smoothed = buffer[-1]
        smoothed = alpha * value + (1 - alpha) * prev_smoothed
        
        return float(smoothed)

    def _determine_status(self,
                         metric_name: str,
                         value: float) -> str:
        """メトリクスのステータス判定"""
        config = self.metric_configs[metric_name]
        
        if value >= config.threshold_critical:
            return 'CRITICAL'
        elif value >= config.threshold_warning:
            return 'WARNING'
        return 'NORMAL'

    def update_baseline(self, 
                       historical_data: Dict[str, List[float]]) -> None:
        """
        ベースラインの更新

        Args:
            historical_data (Dict[str, List[float]]): 過去のメトリクスデータ
        """
        try:
            for metric_name, values in historical_data.items():
                if metric_name in self.metric_configs:
                    # 統計値の計算
                    mean = np.mean(values)
                    std = np.std(values)
                    percentiles = np.percentile(values, [5, 95])
                    
                    self.baseline_stats[metric_name] = {
                        'mean': float(mean),
                        'std': float(std),
                        'percentile_05': float(percentiles[0]),
                        'percentile_95': float(percentiles[1]),
                        'updated_at': datetime.now().isoformat()
                    }
                    
                    # 閾値の自動調整（オプション）
                    config = self.metric_configs[metric_name]
                    config.threshold_warning = mean + 2 * std
                    config.threshold_critical = mean + 3 * std
                    
            logger.info("Baseline statistics updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating baseline: {e}")
            raise

    def get_aggregated_metrics(self, 
                             time_window: int = 300) -> Dict[str, float]:
        """
        メトリクスの集計値を取得

        Args:
            time_window (int): 集計時間枠（秒）

        Returns:
            Dict[str, float]: 集計されたメトリクス
        """
        aggregated = {}
        
        for metric_name, config in self.metric_configs.items():
            buffer = list(self.metric_buffers[metric_name])
            if not buffer:
                continue

            if config.aggregation_method == 'avg':
                value = np.mean(buffer)
            elif config.aggregation_method == 'sum':
                value = np.sum(buffer)
            elif config.aggregation_method == 'max':
                value = np.max(buffer)
            elif config.aggregation_method == 'min':
                value = np.min(buffer)
            else:
                value = np.mean(buffer)  # デフォルト

            aggregated[metric_name] = float(value)

        return aggregated

    def add_metric_config(self,
                         name: str,
                         aggregation_method: str,
                         window_size: int,
                         smoothing_factor: float,
                         threshold_warning: float,
                         threshold_critical: float) -> None:
        """
        新しいメトリクス設定の追加

        Args:
            name (str): メトリクス名
            aggregation_method (str): 集計方法
            window_size (int): ウィンドウサイズ
            smoothing_factor (float): スムージング係数
            threshold_warning (float): 警告閾値
            threshold_critical (float): 危険閾値
        """
        self.metric_configs[name] = MetricConfig(
            name=name,
            aggregation_method=aggregation_method,
            window_size=window_size,
            smoothing_factor=smoothing_factor,
            threshold_warning=threshold_warning,
            threshold_critical=threshold_critical
        )
        
        self.metric_buffers[name] = deque(
            maxlen=max(int(window_size / 10), 100)
        )
        
        logger.info(f"Added new metric configuration: {name}")

    def get_metric_health(self) -> Dict[str, Dict]:
        """
        全メトリクスの健全性レポートを取得

        Returns:
            Dict[str, Dict]: メトリクスの健全性情報
        """
        health_report = {}
        
        for metric_name, config in self.metric_configs.items():
            buffer = list(self.metric_buffers[metric_name])
            if not buffer:
                continue

            current_value = buffer[-1]
            stats = self._calculate_statistics(metric_name)
            trend = self._analyze_trend(metric_name)
            
            health_report[metric_name] = {
                'status': self._determine_status(metric_name, current_value),
                'current_value': current_value,
                'statistics': stats,
                'trend': trend,
                'thresholds': {
                    'warning': config.threshold_warning,
                    'critical': config.threshold_critical
                }
            }

        return health_report
