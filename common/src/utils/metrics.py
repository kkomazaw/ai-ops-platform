from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime
import time
import threading
from functools import wraps
import logging
from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    CollectorRegistry, push_to_gateway,
    start_http_server
)
import psutil
import os
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

@dataclass
class MetricValue:
    """メトリクス値の構造"""
    value: float
    timestamp: datetime
    labels: Dict[str, str]

class MetricsRegistry:
    """メトリクスレジストリ"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """シングルトンパターンの実装"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """メトリクスレジストリの初期化"""
        self.registry = CollectorRegistry()
        self.metrics = {}
        self.custom_metrics = defaultdict(list)

    def counter(self,
                name: str,
                description: str,
                labels: Optional[List[str]] = None) -> Counter:
        """
        カウンターメトリクスの作成

        Args:
            name (str): メトリクス名
            description (str): 説明
            labels (Optional[List[str]]): ラベル

        Returns:
            Counter: カウンターメトリクス
        """
        if name not in self.metrics:
            self.metrics[name] = Counter(
                name,
                description,
                labels or [],
                registry=self.registry
            )
        return self.metrics[name]

    def gauge(self,
             name: str,
             description: str,
             labels: Optional[List[str]] = None) -> Gauge:
        """
        ゲージメトリクスの作成

        Args:
            name (str): メトリクス名
            description (str): 説明
            labels (Optional[List[str]]): ラベル

        Returns:
            Gauge: ゲージメトリクス
        """
        if name not in self.metrics:
            self.metrics[name] = Gauge(
                name,
                description,
                labels or [],
                registry=self.registry
            )
        return self.metrics[name]

    def histogram(self,
                 name: str,
                 description: str,
                 labels: Optional[List[str]] = None,
                 buckets: Optional[List[float]] = None) -> Histogram:
        """
        ヒストグラムメトリクスの作成

        Args:
            name (str): メトリクス名
            description (str): 説明
            labels (Optional[List[str]]): ラベル
            buckets (Optional[List[float]]): バケット定義

        Returns:
            Histogram: ヒストグラムメトリクス
        """
        if name not in self.metrics:
            self.metrics[name] = Histogram(
                name,
                description,
                labels or [],
                buckets=buckets,
                registry=self.registry
            )
        return self.metrics[name]

    def summary(self,
               name: str,
               description: str,
               labels: Optional[List[str]] = None) -> Summary:
        """
        サマリーメトリクスの作成

        Args:
            name (str): メトリクス名
            description (str): 説明
            labels (Optional[List[str]]): ラベル

        Returns:
            Summary: サマリーメトリクス
        """
        if name not in self.metrics:
            self.metrics[name] = Summary(
                name,
                description,
                labels or [],
                registry=self.registry
            )
        return self.metrics[name]

class MetricsCollector:
    """メトリクス収集クラス"""
    def __init__(self,
                 app_name: str,
                 push_gateway_url: Optional[str] = None,
                 export_interval: int = 15):
        """
        初期化

        Args:
            app_name (str): アプリケーション名
            push_gateway_url (Optional[str]): Pushgatewayのエンドポイント
            export_interval (int): エクスポート間隔（秒）
        """
        self.app_name = app_name
        self.push_gateway_url = push_gateway_url
        self.export_interval = export_interval
        self.registry = MetricsRegistry()
        self._setup_default_metrics()
        
        if push_gateway_url:
            self._start_metrics_pusher()

    def _setup_default_metrics(self):
        """デフォルトメトリクスのセットアップ"""
        # システムメトリクス
        self.system_metrics = {
            'cpu_usage': self.registry.gauge(
                'system_cpu_usage',
                'CPU usage percentage'
            ),
            'memory_usage': self.registry.gauge(
                'system_memory_usage',
                'Memory usage percentage'
            ),
            'disk_usage': self.registry.gauge(
                'system_disk_usage',
                'Disk usage percentage'
            )
        }

        # アプリケーションメトリクス
        self.app_metrics = {
            'requests_total': self.registry.counter(
                'app_requests_total',
                'Total number of requests',
                ['method', 'endpoint']
            ),
            'request_duration_seconds': self.registry.histogram(
                'app_request_duration_seconds',
                'Request duration in seconds',
                ['method', 'endpoint'],
                buckets=[.01, .05, .1, .5, 1, 5]
            ),
            'errors_total': self.registry.counter(
                'app_errors_total',
                'Total number of errors',
                ['type']
            )
        }

    def _start_metrics_pusher(self):
        """メトリクスプッシャーの開始"""
        def push_metrics():
            while True:
                try:
                    self._collect_system_metrics()
                    push_to_gateway(
                        self.push_gateway_url,
                        job=self.app_name,
                        registry=self.registry.registry
                    )
                except Exception as e:
                    logger.error(f"Error pushing metrics: {e}")
                time.sleep(self.export_interval)

        thread = threading.Thread(target=push_metrics, daemon=True)
        thread.start()

    def _collect_system_metrics(self):
        """システムメトリクスの収集"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent()
            self.system_metrics['cpu_usage'].set(cpu_percent)

            # メモリ使用率
            memory = psutil.virtual_memory()
            self.system_metrics['memory_usage'].set(memory.percent)

            # ディスク使用率
            disk = psutil.disk_usage('/')
            self.system_metrics['disk_usage'].set(disk.percent)

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

def track_time(metric_name: str):
    """
    実行時間を計測するデコレータ

    Args:
        metric_name (str): メトリクス名
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            registry = MetricsRegistry()
            histogram = registry.histogram(
                f"{metric_name}_duration_seconds",
                f"Duration of {func.__name__} in seconds"
            )
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                histogram.observe(duration)
        return wrapper
    return decorator

def count_calls(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """
    関数呼び出し回数を計測するデコレータ

    Args:
        metric_name (str): メトリクス名
        labels (Optional[Dict[str, str]]): メトリクスラベル
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            registry = MetricsRegistry()
            counter = registry.counter(
                f"{metric_name}_total",
                f"Total number of calls to {func.__name__}",
                list(labels.keys()) if labels else None
            )
            
            try:
                result = func(*args, **kwargs)
                counter.labels(**labels).inc() if labels else counter.inc()
                return result
            except Exception:
                # エラーカウンターの増加
                error_counter = registry.counter(
                    f"{metric_name}_errors_total",
                    f"Total number of errors in {func.__name__}"
                )
                error_counter.inc()
                raise
        return wrapper
    return decorator

class MetricsExporter:
    """メトリクスエクスポーター"""
    def __init__(self, port: int = 8000):
        """
        初期化

        Args:
            port (int): エクスポート用ポート
        """
        self.port = port
        start_http_server(port)
        logger.info(f"Metrics server started on port {port}")

    def export_metrics(self, 
                      format: str = 'prometheus',
                      output_file: Optional[str] = None):
        """
        メトリクスのエクスポート

        Args:
            format (str): 出力フォーマット（'prometheus'または'json'）
            output_file (Optional[str]): 出力ファイルパス
        """
        registry = MetricsRegistry()
        
        if format == 'prometheus':
            metrics_data = ''
            for metric in registry.metrics.values():
                metrics_data += str(metric)
        else:  # json format
            metrics_data = {
                name: [
                    {
                        'value': sample.value,
                        'labels': sample.labels,
                        'timestamp': sample.timestamp
                    }
                    for sample in metric.samples
                ]
                for name, metric in registry.metrics.items()
            }
            metrics_data = json.dumps(metrics_data, indent=2)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(metrics_data)
        
        return metrics_data

# 使用例
if __name__ == "__main__":
    # メトリクスコレクターの初期化
    collector = MetricsCollector(
        "example_app",
        push_gateway_url="http://localhost:9091"
    )

    # メトリクスエクスポーターの開始
    exporter = MetricsExporter(port=8000)

    # デコレータの使用例
    @track_time("example_function")
    @count_calls("example_function", labels={'type': 'test'})
    def example_function():
        time.sleep(0.1)
        return "Done"

    # メトリクスの記録
    for _ in range(5):
        example_function()

    # メトリクスのエクスポート
    exporter.export_metrics(format='json', output_file='metrics.json')

