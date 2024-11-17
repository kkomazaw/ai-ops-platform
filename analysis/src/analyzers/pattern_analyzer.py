from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import json
import re
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class IncidentPattern:
    """インシデントパターンの定義"""
    id: str
    name: str
    description: str
    severity: str
    symptoms: List[str]
    root_causes: List[str]
    solutions: List[str]
    detection_rules: Dict[str, str]
    metrics_patterns: Dict[str, Dict[str, float]]
    occurrence_count: int = 0

@dataclass
class PatternMatch:
    """パターンマッチング結果"""
    pattern_id: str
    confidence: float
    matched_symptoms: List[str]
    matched_metrics: Dict[str, float]
    timestamp: datetime
    incident_id: str
    additional_info: Dict

class PatternAnalyzer:
    def __init__(self):
        """パターン分析エンジンの初期化"""
        self.known_patterns = self._load_patterns()
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.pattern_history: List[PatternMatch] = []
        self.similarity_threshold = 0.7
        self.metric_match_threshold = 0.8

    def _load_patterns(self) -> Dict[str, IncidentPattern]:
        """既知のパターンのロード"""
        try:
            patterns = {
                "db_connection_issue": IncidentPattern(
                    id="PATTERN-001",
                    name="Database Connection Issue",
                    description="Database connectivity problems affecting service availability",
                    severity="high",
                    symptoms=[
                        "connection timeout",
                        "database connection refused",
                        "too many connections",
                        "connection pool exhausted"
                    ],
                    root_causes=[
                        "Connection pool limits reached",
                        "Database server overload",
                        "Network latency issues"
                    ],
                    solutions=[
                        "Increase connection pool size",
                        "Scale up database resources",
                        "Optimize database queries"
                    ],
                    detection_rules={
                        "error_pattern": r"(?i)(connection\s+refused|too\s+many\s+connections)",
                        "frequency": "5/minute"
                    },
                    metrics_patterns={
                        "db_connections": {"min": 80, "max": 100},
                        "response_time": {"min": 1000, "max": float('inf')},
                        "error_rate": {"min": 0.05, "max": 1.0}
                    }
                ),
                "memory_leak": IncidentPattern(
                    id="PATTERN-002",
                    name="Memory Leak",
                    description="Gradual memory consumption increase indicating memory leak",
                    severity="critical",
                    symptoms=[
                        "out of memory error",
                        "memory usage increasing",
                        "garbage collection frequent",
                        "system slowdown"
                    ],
                    root_causes=[
                        "Memory leak in application code",
                        "Resource cleanup issues",
                        "Incorrect memory management"
                    ],
                    solutions=[
                        "Identify memory leak source",
                        "Apply code fixes",
                        "Implement proper resource cleanup"
                    ],
                    detection_rules={
                        "error_pattern": r"(?i)(out\s+of\s+memory|memory\s+leak)",
                        "trend": "increasing"
                    },
                    metrics_patterns={
                        "memory_usage": {"min": 85, "max": 100},
                        "gc_frequency": {"min": 10, "max": float('inf')}
                    }
                ),
                "api_degradation": IncidentPattern(
                    id="PATTERN-003",
                    name="API Performance Degradation",
                    description="API response time degradation affecting service quality",
                    severity="medium",
                    symptoms=[
                        "slow response time",
                        "timeout errors",
                        "increased latency",
                        "request queue building"
                    ],
                    root_causes=[
                        "Backend service overload",
                        "Network congestion",
                        "Resource constraints"
                    ],
                    solutions=[
                        "Scale services horizontally",
                        "Optimize API endpoints",
                        "Implement caching"
                    ],
                    detection_rules={
                        "error_pattern": r"(?i)(timeout|high\s+latency)",
                        "threshold": "response_time > 1000ms"
                    },
                    metrics_patterns={
                        "response_time": {"min": 1000, "max": float('inf')},
                        "error_rate": {"min": 0.02, "max": 1.0},
                        "request_queue": {"min": 100, "max": float('inf')}
                    }
                )
            }
            return patterns
            
        except Exception as e:
            logger.error(f"Error loading patterns: {e}")
            raise

    def analyze_incident(self,
                        incident_data: Dict,
                        metrics: Dict[str, float]) -> List[PatternMatch]:
        """
        インシデントの分析とパターンマッチング

        Args:
            incident_data (Dict): インシデント情報
            metrics (Dict[str, float]): 関連メトリクス

        Returns:
            List[PatternMatch]: マッチしたパターンのリスト
        """
        try:
            matches = []
            
            # 症状のテキスト分析
            incident_text = incident_data.get('description', '') + ' ' + \
                          ' '.join(incident_data.get('symptoms', []))
            
            # 各パターンとの照合
            for pattern_id, pattern in self.known_patterns.items():
                # テキストベースのマッチング
                symptom_match = self._match_symptoms(incident_text, pattern)
                
                # メトリクスベースのマッチング
                metrics_match = self._match_metrics(metrics, pattern.metrics_patterns)
                
                # 総合的な信頼度の計算
                confidence = (symptom_match['score'] + metrics_match['score']) / 2
                
                if confidence >= self.similarity_threshold:
                    match = PatternMatch(
                        pattern_id=pattern_id,
                        confidence=confidence,
                        matched_symptoms=symptom_match['matched_symptoms'],
                        matched_metrics=metrics_match['matched_metrics'],
                        timestamp=datetime.now(),
                        incident_id=incident_data.get('id', ''),
                        additional_info={
                            'symptom_score': symptom_match['score'],
                            'metrics_score': metrics_match['score']
                        }
                    )
                    matches.append(match)
            
            # パターンの更新
            self._update_pattern_statistics(matches)
            
            return sorted(matches, key=lambda x: x.confidence, reverse=True)
            
        except Exception as e:
            logger.error(f"Error analyzing incident: {e}")
            raise

    def _match_symptoms(self, incident_text: str, pattern: IncidentPattern) -> Dict:
        """症状のマッチング分析"""
        matched_symptoms = []
        total_symptoms = len(pattern.symptoms)
        matched_count = 0

        # 正規表現パターンのチェック
        if 'error_pattern' in pattern.detection_rules:
            if re.search(pattern.detection_rules['error_pattern'], incident_text, re.I):
                matched_count += 1

        # 症状キーワードのチェック
        for symptom in pattern.symptoms:
            if symptom.lower() in incident_text.lower():
                matched_symptoms.append(symptom)
                matched_count += 1

        score = matched_count / total_symptoms if total_symptoms > 0 else 0

        return {
            'score': score,
            'matched_symptoms': matched_symptoms
        }

    def _match_metrics(self, 
                      current_metrics: Dict[str, float],
                      pattern_metrics: Dict[str, Dict[str, float]]) -> Dict:
        """メトリクスパターンのマッチング"""
        matched_metrics = {}
        total_metrics = len(pattern_metrics)
        matched_count = 0

        for metric_name, thresholds in pattern_metrics.items():
            if metric_name in current_metrics:
                value = current_metrics[metric_name]
                if thresholds['min'] <= value <= thresholds['max']:
                    matched_metrics[metric_name] = value
                    matched_count += 1

        score = matched_count / total_metrics if total_metrics > 0 else 0

        return {
            'score': score,
            'matched_metrics': matched_metrics
        }

    def _update_pattern_statistics(self, matches: List[PatternMatch]) -> None:
        """パターン統計の更新"""
        for match in matches:
            if match.pattern_id in self.known_patterns:
                pattern = self.known_patterns[match.pattern_id]
                pattern.occurrence_count += 1
                self.pattern_history.append(match)

    def add_new_pattern(self, pattern: IncidentPattern) -> None:
        """新しいパターンの追加"""
        if pattern.id in self.known_patterns:
            raise ValueError(f"Pattern with ID {pattern.id} already exists")
            
        self.known_patterns[pattern.id] = pattern
        logger.info(f"Added new pattern: {pattern.id}")

    def get_pattern_statistics(self) -> Dict:
        """パターン統計の取得"""
        stats = {
            'total_patterns': len(self.known_patterns),
            'total_matches': len(self.pattern_history),
            'pattern_frequencies': {},
            'recent_matches': []
        }

        # パターンごとの発生頻度
        for pattern_id, pattern in self.known_patterns.items():
            stats['pattern_frequencies'][pattern_id] = {
                'name': pattern.name,
                'count': pattern.occurrence_count,
                'severity': pattern.severity
            }

        # 最近のマッチング
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_matches = [
            match for match in self.pattern_history
            if match.timestamp >= recent_cutoff
        ]
        
        stats['recent_matches'] = [
            {
                'pattern_id': match.pattern_id,
                'incident_id': match.incident_id,
                'confidence': match.confidence,
                'timestamp': match.timestamp.isoformat()
            }
            for match in recent_matches
        ]

        return stats

    def suggest_solutions(self, matches: List[PatternMatch]) -> List[Dict]:
        """解決策の提案"""
        suggestions = []
        
        for match in matches:
            if match.pattern_id in self.known_patterns:
                pattern = self.known_patterns[match.pattern_id]
                suggestion = {
                    'pattern_id': match.pattern_id,
                    'pattern_name': pattern.name,
                    'confidence': match.confidence,
                    'solutions': pattern.solutions,
                    'root_causes': pattern.root_causes
                }
                suggestions.append(suggestion)

        return sorted(suggestions, key=lambda x: x['confidence'], reverse=True)

    def export_patterns(self, filepath: str) -> None:
        """パターン定義のエクスポート"""
        try:
            patterns_data = {
                pattern_id: {
                    'id': pattern.id,
                    'name': pattern.name,
                    'description': pattern.description,
                    'severity': pattern.severity,
                    'symptoms': pattern.symptoms,
                    'root_causes': pattern.root_causes,
                    'solutions': pattern.solutions,
                    'detection_rules': pattern.detection_rules,
                    'metrics_patterns': pattern.metrics_patterns,
                    'occurrence_count': pattern.occurrence_count
                }
                for pattern_id, pattern in self.known_patterns.items()
            }

            with open(filepath, 'w') as f:
                json.dump(patterns_data, f, indent=2)
                
            logger.info(f"Exported patterns to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting patterns: {e}")
            raise

    def import_patterns(self, filepath: str) -> None:
        """パターン定義のインポート"""
        try:
            with open(filepath, 'r') as f:
                patterns_data = json.load(f)

            for pattern_id, data in patterns_data.items():
                pattern = IncidentPattern(**data)
                self.known_patterns[pattern_id] = pattern

            logger.info(f"Imported patterns from {filepath}")
            
        except Exception as e:
            logger.error(f"Error importing patterns: {e}")
            raise
