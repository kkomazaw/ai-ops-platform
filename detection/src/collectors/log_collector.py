import re
from typing import Dict, List, Optional, Set
import logging
from datetime import datetime, timedelta
import elasticsearch
from elasticsearch import Elasticsearch
import json
import pandas as pd
from dataclasses import dataclass
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LogPattern:
    """ログパターン定義"""
    pattern: str
    severity: str
    category: str
    description: str

class LogCollector:
    def __init__(self,
                 es_hosts: List[str],
                 index_pattern: str = "logs-*",
                 basic_auth: Optional[Dict[str, str]] = None):
        """
        ログ収集クラス

        Args:
            es_hosts (List[str]): Elasticsearchホスト
            index_pattern (str): インデックスパターン
            basic_auth (Optional[Dict[str, str]]): 認証情報
        """
        self.es = Elasticsearch(
            es_hosts,
            basic_auth=(basic_auth['username'], basic_auth['password']) if basic_auth else None,
            verify_certs=False
        )
        self.index_pattern = index_pattern
        self.log_patterns = self._initialize_log_patterns()
        self.error_patterns = self._initialize_error_patterns()

    def _initialize_log_patterns(self) -> Dict[str, LogPattern]:
        """既知のログパターンの初期化"""
        return {
            'db_connection_error': LogPattern(
                pattern=r"(?i)(connection\s+refused|too\s+many\s+connections|connection\s+timed?\s*out)",
                severity="ERROR",
                category="database",
                description="Database connection issues"
            ),
            'memory_error': LogPattern(
                pattern=r"(?i)(out\s+of\s+memory|memory\s+exhausted|cannot\s+allocate\s+memory)",
                severity="CRITICAL",
                category="system",
                description="Memory resource issues"
            ),
            'api_error': LogPattern(
                pattern=r"(?i)(5\d{2}\s+error|api\s+timeout|api\s+failed)",
                severity="ERROR",
                category="api",
                description="API related errors"
            ),
            'security_warning': LogPattern(
                pattern=r"(?i)(unauthorized|forbidden|invalid\s+token|security\s+breach)",
                severity="WARNING",
                category="security",
                description="Security related warnings"
            ),
            'performance_warning': LogPattern(
                pattern=r"(?i)(slow\s+query|high\s+latency|timeout|performance\s+degradation)",
                severity="WARNING",
                category="performance",
                description="Performance issues"
            )
        }

    def _initialize_error_patterns(self) -> Dict[str, Set[str]]:
        """エラーパターンのカテゴリ別初期化"""
        return {
            'database': {
                'deadlock', 'lock timeout', 'connection refused',
                'too many connections', 'database is full'
            },
            'memory': {
                'out of memory', 'memory exhausted', 'cannot allocate memory',
                'memory quota exceeded', 'kill process or sacrifice child'
            },
            'network': {
                'connection refused', 'connection timeout', 'network unreachable',
                'no route to host', 'connection reset'
            },
            'system': {
                'system overload', 'high cpu usage', 'disk space low',
                'resource exhausted', 'process killed'
            },
            'application': {
                'null pointer', 'invalid argument', 'undefined index',
                'division by zero', 'stack overflow'
            }
        }

    def collect_recent_logs(self, 
                          time_range_minutes: int = 5,
                          max_logs: int = 1000) -> List[Dict]:
        """
        最近のログを収集

        Args:
            time_range_minutes (int): 収集する時間範囲（分）
            max_logs (int): 最大ログ数

        Returns:
            List[Dict]: 収集したログのリスト
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=time_range_minutes)

            query = {
                "bool": {
                    "must": [
                        {
                            "range": {
                                "@timestamp": {
                                    "gte": start_time.isoformat(),
                                    "lte": end_time.isoformat()
                                }
                            }
                        }
                    ]
                }
            }

            response = self.es.search(
                index=self.index_pattern,
                body={
                    "query": query,
                    "sort": [{"@timestamp": "desc"}],
                    "size": max_logs
                }
            )

            logs = []
            for hit in response['hits']['hits']:
                log_entry = hit['_source']
                log_entry['_id'] = hit['_id']
                logs.append(log_entry)

            return logs

        except Exception as e:
            logger.error(f"Error collecting recent logs: {e}")
            raise

    def analyze_logs(self, logs: List[Dict]) -> Dict:
        """
        ログの分析を実行

        Args:
            logs (List[Dict]): 分析対象のログ

        Returns:
            Dict: 分析結果
        """
        analysis_result = {
            'pattern_matches': defaultdict(list),
            'error_counts': defaultdict(int),
            'severity_counts': defaultdict(int),
            'temporal_patterns': [],
            'anomalies': []
        }

        try:
            # パターンマッチング
            for log in logs:
                message = log.get('message', '')
                
                # 既知のパターンチェック
                for pattern_name, pattern in self.log_patterns.items():
                    if re.search(pattern.pattern, message, re.IGNORECASE):
                        analysis_result['pattern_matches'][pattern_name].append({
                            'timestamp': log.get('@timestamp'),
                            'message': message,
                            'severity': pattern.severity,
                            'category': pattern.category
                        })
                        analysis_result['severity_counts'][pattern.severity] += 1

                # エラーカテゴリの分類
                for category, patterns in self.error_patterns.items():
                    for pattern in patterns:
                        if pattern.lower() in message.lower():
                            analysis_result['error_counts'][category] += 1

            # 時系列パターンの検出
            temporal_patterns = self._detect_temporal_patterns(logs)
            analysis_result['temporal_patterns'] = temporal_patterns

            # 異常パターンの検出
            anomalies = self._detect_anomalies(logs)
            analysis_result['anomalies'] = anomalies

            return analysis_result

        except Exception as e:
            logger.error(f"Error analyzing logs: {e}")
            raise

    def _detect_temporal_patterns(self, logs: List[Dict]) -> List[Dict]:
        """時系列パターンの検出"""
        patterns = []
        
        try:
            # タイムスタンプでソート
            sorted_logs = sorted(logs, key=lambda x: x.get('@timestamp', ''))
            
            # 時間間隔ごとのエラー数をカウント
            time_windows = defaultdict(int)
            window_size = timedelta(minutes=1)
            
            for log in sorted_logs:
                timestamp = datetime.fromisoformat(log.get('@timestamp', '').replace('Z', '+00:00'))
                window_key = timestamp.replace(second=0, microsecond=0)
                time_windows[window_key] += 1

            # 急増を検出
            for window_time, count in time_windows.items():
                if count > 10:  # 閾値は要調整
                    patterns.append({
                        'type': 'spike',
                        'timestamp': window_time.isoformat(),
                        'count': count,
                        'description': f"Log volume spike detected: {count} logs/minute"
                    })

            return patterns

        except Exception as e:
            logger.error(f"Error detecting temporal patterns: {e}")
            return []

    def _detect_anomalies(self, logs: List[Dict]) -> List[Dict]:
        """異常パターンの検出"""
        anomalies = []
        
        try:
            error_sequence = []
            current_sequence = []

            for log in logs:
                message = log.get('message', '')
                
                # 重大なエラーパターンの検出
                for pattern_name, pattern in self.log_patterns.items():
                    if pattern.severity == "CRITICAL" and re.search(pattern.pattern, message, re.IGNORECASE):
                        anomalies.append({
                            'type': 'critical_error',
                            'pattern': pattern_name,
                            'timestamp': log.get('@timestamp'),
                            'message': message,
                            'description': pattern.description
                        })

                # エラーの連続性を検出
                if any(error in message.lower() for category in self.error_patterns.values() for error in category):
                    current_sequence.append(log)
                else:
                    if len(current_sequence) >= 3:  # 連続エラーの閾値
                        error_sequence.append({
                            'start_time': current_sequence[0].get('@timestamp'),
                            'end_time': current_sequence[-1].get('@timestamp'),
                            'count': len(current_sequence),
                            'description': "Consecutive error sequence detected"
                        })
                    current_sequence = []

            return anomalies + error_sequence

        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return []

    def get_log_summary(self, time_range_hours: int = 24) -> Dict:
        """
        ログサマリーの生成

        Args:
            time_range_hours (int): 対象期間（時間）

        Returns:
            Dict: ログサマリー情報
        """
        try:
            logs = self.collect_recent_logs(time_range_minutes=time_range_hours * 60)
            analysis = self.analyze_logs(logs)

            return {
                'total_logs': len(logs),
                'error_distribution': dict(analysis['error_counts']),
                'severity_distribution': dict(analysis['severity_counts']),
                'pattern_matches': {k: len(v) for k, v in analysis['pattern_matches'].items()},
                'anomaly_count': len(analysis['anomalies']),
                'temporal_patterns': analysis['temporal_patterns']
            }

        except Exception as e:
            logger.error(f"Error generating log summary: {e}")
            raise

    def export_logs(self, 
                   output_format: str = 'json',
                   filepath: str = None) -> Optional[str]:
        """
        ログのエクスポート

        Args:
            output_format (str): 出力フォーマット（json/csv）
            filepath (str): 出力ファイルパス

        Returns:
            Optional[str]: エクスポートされたファイルパス
        """
        try:
            logs = self.collect_recent_logs(time_range_minutes=60)
            
            if output_format.lower() == 'json':
                output_data = json.dumps(logs, indent=2)
                file_extension = 'json'
            elif output_format.lower() == 'csv':
                df = pd.DataFrame(logs)
                output_data = df.to_csv(index=False)
                file_extension = 'csv'
            else:
                raise ValueError(f"Unsupported output format: {output_format}")

            if filepath:
                output_path = filepath
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f"logs_export_{timestamp}.{file_extension}"

            with open(output_path, 'w') as f:
                f.write(output_data)

            return output_path

        except Exception as e:
            logger.error(f"Error exporting logs: {e}")
            raise
