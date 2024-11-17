from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import json
import re
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from textblob import TextBlob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LogEntry:
    """ログエントリの構造"""
    timestamp: datetime
    level: str
    service: str
    message: str
    trace_id: Optional[str]
    additional_info: Dict

@dataclass
class LogPattern:
    """ログパターンの構造"""
    pattern_id: str
    regex: str
    description: str
    severity: str
    category: str
    related_services: List[str]

@dataclass
class LogAnalysisResult:
    """分析結果の構造"""
    patterns: List[Dict]
    anomalies: List[Dict]
    error_clusters: List[Dict]
    service_health: Dict[str, str]
    temporal_patterns: List[Dict]
    summary: str

class LogAnalyzer:
    def __init__(self):
        """ログ分析エンジンの初期化"""
        self.patterns = self._initialize_patterns()
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.error_keywords = self._load_error_keywords()
        self.service_dependencies = self._load_service_dependencies()
        self.severity_weights = {
            'CRITICAL': 4,
            'ERROR': 3,
            'WARNING': 2,
            'INFO': 1
        }

    def _initialize_patterns(self) -> List[LogPattern]:
        """ログパターンの初期化"""
        return [
            LogPattern(
                pattern_id="DB-001",
                regex=r"(?i)(connection\s+refused|too\s+many\s+connections)",
                description="Database connection issues",
                severity="ERROR",
                category="database",
                related_services=["db-service", "user-service"]
            ),
            LogPattern(
                pattern_id="MEM-001",
                regex=r"(?i)(out\s+of\s+memory|memory\s+leak)",
                description="Memory related issues",
                severity="CRITICAL",
                category="resource",
                related_services=["all"]
            ),
            LogPattern(
                pattern_id="API-001",
                regex=r"(?i)(timeout|request\s+failed|5\d{2}\s+error)",
                description="API call failures",
                severity="ERROR",
                category="api",
                related_services=["api-gateway", "auth-service"]
            ),
            LogPattern(
                pattern_id="SEC-001",
                regex=r"(?i)(unauthorized|forbidden|invalid\s+token)",
                description="Security issues",
                severity="WARNING",
                category="security",
                related_services=["auth-service"]
            )
        ]

    def _load_error_keywords(self) -> Dict[str, List[str]]:
        """エラーキーワードの読み込み"""
        return {
            'critical': [
                'crash', 'fatal', 'emergency', 'kernel', 'deadlock',
                'corruption', 'breach', 'outage'
            ],
            'error': [
                'exception', 'failed', 'error', 'invalid', 'timeout',
                'unavailable', 'denied'
            ],
            'warning': [
                'warning', 'deprecated', 'high load', 'retry',
                'slow', 'delayed'
            ]
        }

    def _load_service_dependencies(self) -> Dict[str, List[str]]:
        """サービス依存関係の読み込み"""
        return {
            'api-gateway': ['auth-service', 'user-service'],
            'user-service': ['db-service', 'cache-service'],
            'auth-service': ['db-service', 'token-service'],
            'payment-service': ['db-service', 'external-payment-api']
        }

    def analyze_logs(self, logs: List[LogEntry]) -> LogAnalysisResult:
        """
        ログの総合分析を実行

        Args:
            logs (List[LogEntry]): 分析対象のログエントリリスト

        Returns:
            LogAnalysisResult: 分析結果
        """
        try:
            # パターンマッチング
            pattern_matches = self._find_patterns(logs)
            
            # 異常検知
            anomalies = self._detect_anomalies(logs)
            
            # エラークラスタリング
            error_clusters = self._cluster_errors(logs)
            
            # サービス健全性評価
            service_health = self._evaluate_service_health(logs)
            
            # 時系列パターン分析
            temporal_patterns = self._analyze_temporal_patterns(logs)
            
            # サマリー生成
            summary = self._generate_summary(
                pattern_matches,
                anomalies,
                error_clusters,
                service_health
            )
            
            return LogAnalysisResult(
                patterns=pattern_matches,
                anomalies=anomalies,
                error_clusters=error_clusters,
                service_health=service_health,
                temporal_patterns=temporal_patterns,
                summary=summary
            )
            
        except Exception as e:
            logger.error(f"Error analyzing logs: {e}")
            raise

    def _find_patterns(self, logs: List[LogEntry]) -> List[Dict]:
        """パターンマッチングの実行"""
        matches = []
        
        for log in logs:
            for pattern in self.patterns:
                if re.search(pattern.regex, log.message):
                    matches.append({
                        'pattern_id': pattern.pattern_id,
                        'timestamp': log.timestamp,
                        'service': log.service,
                        'message': log.message,
                        'severity': pattern.severity,
                        'category': pattern.category
                    })
        
        return self._aggregate_pattern_matches(matches)

    def _detect_anomalies(self, logs: List[LogEntry]) -> List[Dict]:
        """異常検知の実行"""
        anomalies = []
        
        # 時間枠ごとのログ頻度分析
        frequency_anomalies = self._analyze_log_frequency(logs)
        anomalies.extend(frequency_anomalies)
        
        # エラー率の分析
        error_anomalies = self._analyze_error_rates(logs)
        anomalies.extend(error_anomalies)
        
        # サービス間の異常な依存関係
        dependency_anomalies = self._analyze_service_dependencies(logs)
        anomalies.extend(dependency_anomalies)
        
        return anomalies

    def _cluster_errors(self, logs: List[LogEntry]) -> List[Dict]:
        """エラーメッセージのクラスタリング"""
        # エラーログの抽出
        error_logs = [
            log for log in logs
            if log.level in ['ERROR', 'CRITICAL']
        ]
        
        if not error_logs:
            return []
            
        # テキストのベクトル化
        messages = [log.message for log in error_logs]
        vectors = self.vectorizer.fit_transform(messages)
        
        # クラスタリングの実行
        clustering = DBSCAN(
            eps=0.3,
            min_samples=2,
            metric='cosine'
        ).fit(vectors)
        
        # クラスタの集計
        clusters = defaultdict(list)
        for i, label in enumerate(clustering.labels_):
            if label != -1:  # ノイズを除外
                clusters[label].append(error_logs[i])
        
        return [
            {
                'cluster_id': f"C{label}",
                'size': len(cluster_logs),
                'representative_message': self._get_representative_message(cluster_logs),
                'services': list(set(log.service for log in cluster_logs)),
                'time_range': {
                    'start': min(log.timestamp for log in cluster_logs),
                    'end': max(log.timestamp for log in cluster_logs)
                }
            }
            for label, cluster_logs in clusters.items()
        ]

    def _evaluate_service_health(self, logs: List[LogEntry]) -> Dict[str, str]:
        """サービス健全性の評価"""
        service_stats = defaultdict(lambda: {
            'error_count': 0,
            'total_logs': 0,
            'severity_score': 0
        })
        
        # サービスごとの統計収集
        for log in logs:
            stats = service_stats[log.service]
            stats['total_logs'] += 1
            
            if log.level in ['ERROR', 'CRITICAL']:
                stats['error_count'] += 1
            
            stats['severity_score'] += self.severity_weights.get(log.level, 0)
        
        # 健全性評価
        health_status = {}
        for service, stats in service_stats.items():
            error_rate = stats['error_count'] / stats['total_logs']
            severity_avg = stats['severity_score'] / stats['total_logs']
            
            if error_rate > 0.1 or severity_avg > 3:
                health_status[service] = 'critical'
            elif error_rate > 0.05 or severity_avg > 2:
                health_status[service] = 'warning'
            else:
                health_status[service] = 'healthy'
        
        return health_status

    def _analyze_temporal_patterns(self, logs: List[LogEntry]) -> List[Dict]:
        """時系列パターンの分析"""
        # 時間枠ごとのログ集計
        df = pd.DataFrame([
            {
                'timestamp': log.timestamp,
                'service': log.service,
                'level': log.level
            }
            for log in logs
        ])
        
        if df.empty:
            return []
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # 5分間隔でリサンプリング
        resampled = df.groupby(['service', 'level']).resample('5T').size()
        
        patterns = []
        for (service, level), series in resampled.groupby(['service', 'level']):
            # 急激な変化の検出
            changes = series.pct_change()
            spikes = changes[changes > 1.0]  # 100%以上の増加
            
            if not spikes.empty:
                patterns.append({
                    'type': 'spike',
                    'service': service,
                    'level': level,
                    'timestamps': spikes.index.tolist(),
                    'magnitude': spikes.values.tolist()
                })
        
        return patterns

    def _generate_summary(self,
                         patterns: List[Dict],
                         anomalies: List[Dict],
                         error_clusters: List[Dict],
                         service_health: Dict[str, str]) -> str:
        """分析結果のサマリー生成"""
        critical_services = [
            service for service, status in service_health.items()
            if status == 'critical'
        ]
        
        summary_parts = []
        
        if critical_services:
            summary_parts.append(
                f"Critical issues detected in services: {', '.join(critical_services)}"
            )
        
        if patterns:
            top_patterns = sorted(
                patterns,
                key=lambda x: len(x.get('occurrences', [])),
                reverse=True
            )[:3]
            summary_parts.append(
                "Top patterns: " + 
                "; ".join(f"{p['pattern_id']}: {p['description']}" for p in top_patterns)
            )
        
        if error_clusters:
            summary_parts.append(
                f"Identified {len(error_clusters)} distinct error patterns"
            )
        
        if anomalies:
            summary_parts.append(
                f"Detected {len(anomalies)} anomalous behaviors"
            )
        
        return " ".join(summary_parts)

    def _aggregate_pattern_matches(self, matches: List[Dict]) -> List[Dict]:
        """パターンマッチの集計"""
        aggregated = defaultdict(lambda: {
            'occurrences': [],
            'services': set(),
            'description': '',
            'severity': '',
            'category': ''
        })
        
        for match in matches:
            pattern_data = aggregated[match['pattern_id']]
            pattern_data['occurrences'].append({
                'timestamp': match['timestamp'],
                'message': match['message']
            })
            pattern_data['services'].add(match['service'])
            pattern_data['description'] = match.get('description', '')
            pattern_data['severity'] = match.get('severity', '')
            pattern_data['category'] = match.get('category', '')
        
        return [
            {
                'pattern_id': pattern_id,
                'occurrences': data['occurrences'],
                'services': list(data['services']),
                'description': data['description'],
                'severity': data['severity'],
                'category': data['category'],
                'count': len(data['occurrences'])
            }
            for pattern_id, data in aggregated.items()
        ]

    def _get_representative_message(self, logs: List[LogEntry]) -> str:
        """クラスタの代表的なメッセージを取得"""
        messages = [log.message for log in logs]
        
        # TF-IDFベースで最も代表的なメッセージを選択
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(messages)
        centroid = vectors.mean(axis=0)
        
        # 中心に最も近いメッセージを選択
        similarities = cosine_similarity(vectors, centroid)
        representative_idx = np.argmax(similarities)
        
        return messages[representative_idx]

    def add_pattern(self, pattern: LogPattern) -> None:
        """新しいパターンの追加"""
        self.patterns.append(pattern)
        logger.info(f"Added new pattern: {pattern.pattern_id}")

    def export_analysis(self, 
                       result: LogAnalysisResult,
                       filepath: str) -> None:
        """分析結果のエクスポート"""
        try:
            output = {
                'summary': result.summary,
                'patterns': result.patterns,
                'anomalies': result.anomalies,
                'error_clusters': result.error_clusters,
                'service_health': result.service_health,
                'temporal_patterns': result.temporal_patterns,
                'generated_at': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(output, f, indent=2, default=str)
                
            logger.info(f"Analysis results exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting analysis results: {e}")
            raise
