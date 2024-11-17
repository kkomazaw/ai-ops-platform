import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import networkx as nx
from collections import defaultdict
import re
import logging
import json
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RootCauseAnalyzer:
    def __init__(self):
        """原因分析システムの初期化"""
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = TFBertModel.from_pretrained('bert-base-uncased')
        self.rf_classifier = RandomForestClassifier(n_estimators=100)
        self.scaler = StandardScaler()
        self.known_patterns = self._load_known_patterns()
        self.dependency_graph = self._build_dependency_graph()
        
    def _load_known_patterns(self):
        """既知の障害パターンデータベースの読み込み"""
        return {
            'database_connection_limit': {
                'symptoms': ['high_active_users', 'database_errors', 'connection_timeout'],
                'metrics_pattern': {
                    'db_connections': 'high',
                    'transaction_time': 'high',
                    'error_rate': 'increasing'
                },
                'log_patterns': [
                    r"connection refused",
                    r"too many connections",
                    r"connection pool exhausted"
                ]
            },
            'memory_leak': {
                'symptoms': ['high_memory_usage', 'slow_response', 'oom_killer'],
                'metrics_pattern': {
                    'memory_usage': 'increasing',
                    'transaction_time': 'increasing',
                    'error_rate': 'normal'
                },
                'log_patterns': [
                    r"OutOfMemoryError",
                    r"Memory quota exceeded",
                    r"Kill process or sacrifice child"
                ]
            },
            'deadlock': {
                'symptoms': ['transaction_timeout', 'database_errors'],
                'metrics_pattern': {
                    'transaction_time': 'very_high',
                    'error_rate': 'high',
                    'db_connections': 'high'
                },
                'log_patterns': [
                    r"Deadlock found",
                    r"Lock wait timeout exceeded",
                    r"Transaction.*rolled back"
                ]
            }
        }

    def _build_dependency_graph(self):
        """システムコンポーネント間の依存関係グラフの構築"""
        G = nx.DiGraph()
        
        # ノードの追加
        components = [
            'web_server', 'application_server', 'database',
            'cache', 'message_queue', 'storage',
            'load_balancer', 'authentication_service'
        ]
        G.add_nodes_from(components)
        
        # エッジの追加（依存関係）
        dependencies = [
            ('web_server', 'load_balancer'),
            ('load_balancer', 'application_server'),
            ('application_server', 'database'),
            ('application_server', 'cache'),
            ('application_server', 'message_queue'),
            ('application_server', 'authentication_service'),
            ('database', 'storage')
        ]
        G.add_edges_from(dependencies)
        
        return G

    def analyze_root_cause(self, anomaly_data, system_logs, metrics_history):
        """
        総合的な原因分析の実行
        
        Parameters:
        anomaly_data (dict): 異常検知結果
        system_logs (list): システムログデータ
        metrics_history (dict): 過去のメトリクスデータ
        
        Returns:
        dict: 分析結果と推奨アクション
        """
        try:
            # 1. パターンマッチング分析
            pattern_analysis = self._pattern_analysis(anomaly_data, system_logs)
            
            # 2. メトリクス相関分析
            metrics_analysis = self._analyze_metrics_correlation(metrics_history)
            
            # 3. ログ分析
            log_analysis = self._analyze_logs(system_logs)
            
            # 4. 依存関係分析
            dependency_analysis = self._analyze_dependencies(anomaly_data)
            
            # 5. 結果の統合と根本原因の特定
            root_cause = self._integrate_analysis_results(
                pattern_analysis,
                metrics_analysis,
                log_analysis,
                dependency_analysis
            )
            
            # 6. 推奨アクションの生成
            recommended_actions = self._generate_recommendations(root_cause)
            
            return {
                'root_cause': root_cause,
                'confidence_score': root_cause['confidence'],
                'recommended_actions': recommended_actions,
                'analysis_details': {
                    'pattern_analysis': pattern_analysis,
                    'metrics_analysis': metrics_analysis,
                    'log_analysis': log_analysis,
                    'dependency_analysis': dependency_analysis
                }
            }
            
        except Exception as e:
            logger.error(f"Error in root cause analysis: {e}")
            raise

    def _pattern_analysis(self, anomaly_data, system_logs):
        """
        既知のパターンとの照合による分析
        
        Parameters:
        anomaly_data (dict): 異常検知データ
        system_logs (list): システムログ
        
        Returns:
        dict: パターン分析結果
        """
        pattern_matches = []
        
        for pattern_name, pattern_info in self.known_patterns.items():
            match_score = 0
            total_checks = 0
            
            # メトリクスパターンの照合
            for metric, expected in pattern_info['metrics_pattern'].items():
                if metric in anomaly_data:
                    total_checks += 1
                    if self._match_metric_pattern(anomaly_data[metric], expected):
                        match_score += 1
            
            # ログパターンの照合
            for log_pattern in pattern_info['log_patterns']:
                total_checks += 1
                if any(re.search(log_pattern, log) for log in system_logs):
                    match_score += 1
            
            # 症状の照合
            for symptom in pattern_info['symptoms']:
                total_checks += 1
                if self._check_symptom(symptom, anomaly_data, system_logs):
                    match_score += 1
            
            if total_checks > 0:
                confidence = match_score / total_checks
                if confidence > 0.6:  # 60%以上の一致で有力なパターンとして記録
                    pattern_matches.append({
                        'pattern': pattern_name,
                        'confidence': confidence,
                        'matching_metrics': match_score,
                        'total_checks': total_checks
                    })
        
        return {
            'matched_patterns': pattern_matches,
            'best_match': max(pattern_matches, key=lambda x: x['confidence']) if pattern_matches else None
        }

    def _analyze_metrics_correlation(self, metrics_history):
        """
        メトリクス間の相関関係分析
        
        Parameters:
        metrics_history (dict): 過去のメトリクスデータ
        
        Returns:
        dict: 相関分析結果
        """
        df = pd.DataFrame(metrics_history)
        
        # 相関行列の計算
        correlation_matrix = df.corr()
        
        # 強い相関関係の抽出
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > 0.7:  # 強い相関の閾値
                    strong_correlations.append({
                        'metrics': (correlation_matrix.columns[i], correlation_matrix.columns[j]),
                        'correlation': corr
                    })
        
        # 異常値の時系列分析
        anomaly_sequence = self._analyze_anomaly_sequence(df)
        
        return {
            'correlations': strong_correlations,
            'anomaly_sequence': anomaly_sequence,
            'correlation_matrix': correlation_matrix.to_dict()
        }

    def _analyze_logs(self, system_logs):
        """
        システムログの詳細分析
        
        Parameters:
        system_logs (list): システムログデータ
        
        Returns:
        dict: ログ分析結果
        """
        # ログのBERT埋め込み処理
        encoded_logs = self._encode_logs(system_logs)
        
        # クラスタリングによるログパターンの検出
        log_clusters = self._cluster_logs(encoded_logs)
        
        # エラーパターンの抽出
        error_patterns = self._extract_error_patterns(system_logs)
        
        # 時系列での異常イベントの特定
        temporal_analysis = self._analyze_temporal_patterns(system_logs)
        
        return {
            'log_clusters': log_clusters,
            'error_patterns': error_patterns,
            'temporal_analysis': temporal_analysis,
            'critical_events': self._identify_critical_events(system_logs)
        }

    def _analyze_dependencies(self, anomaly_data):
        """
        コンポーネント間の依存関係分析
        
        Parameters:
        anomaly_data (dict): 異常検知データ
        
        Returns:
        dict: 依存関係分析結果
        """
        affected_components = set()
        
        # 異常が検出されたコンポーネントの特定
        for metric, data in anomaly_data.items():
            if data['severity'] in ['WARNING', 'CRITICAL']:
                component = self._map_metric_to_component(metric)
                affected_components.add(component)
        
        # 影響範囲の分析
        impact_graph = self._analyze_impact_propagation(affected_components)
        
        # 重要度に基づくコンポーネントのランキング
        component_ranking = self._rank_components_by_importance(impact_graph)
        
        return {
            'affected_components': list(affected_components),
            'impact_graph': impact_graph,
            'component_ranking': component_ranking
        }

    def _integrate_analysis_results(self, pattern_analysis, metrics_analysis, 
                                  log_analysis, dependency_analysis):
        """
        各分析結果の統合と根本原因の特定
        
        Parameters:
        pattern_analysis (dict): パターン分析結果
        metrics_analysis (dict): メトリクス分析結果
        log_analysis (dict): ログ分析結果
        dependency_analysis (dict): 依存関係分析結果
        
        Returns:
        dict: 統合された分析結果
        """
        causes = []
        
        # パターンマッチングからの候補
        if pattern_analysis['best_match']:
            causes.append({
                'cause': pattern_analysis['best_match']['pattern'],
                'confidence': pattern_analysis['best_match']['confidence'],
                'evidence': 'pattern_matching'
            })
        
        # メトリクス相関からの候補
        for correlation in metrics_analysis['correlations']:
            if correlation['correlation'] > 0.9:  # 非常に強い相関
                causes.append({
                    'cause': f"Strong correlation between {correlation['metrics'][0]} and {correlation['metrics'][1]}",
                    'confidence': abs(correlation['correlation']),
                    'evidence': 'metric_correlation'
                })
        
        # ログ分析からの候補
        for event in log_analysis['critical_events']:
            causes.append({
                'cause': event['description'],
                'confidence': event['severity_score'],
                'evidence': 'log_analysis'
            })
        
        # 依存関係分析からの候補
        for component in dependency_analysis['component_ranking'][:2]:  # 上位2つのコンポーネント
            causes.append({
                'cause': f"Critical component failure: {component['name']}",
                'confidence': component['importance_score'],
                'evidence': 'dependency_analysis'
            })
        
        # 最も可能性の高い原因の選定
        primary_cause = max(causes, key=lambda x: x['confidence'])
        
        return {
            'primary_cause': primary_cause['cause'],
            'confidence': primary_cause['confidence'],
            'evidence_type': primary_cause['evidence'],
            'supporting_causes': causes,
            'affected_components': dependency_analysis['affected_components']
        }

    def _generate_recommendations(self, root_cause):
        """
        特定された原因に基づく推奨アクションの生成
        
        Parameters:
        root_cause (dict): 特定された根本原因
        
        Returns:
        list: 推奨アクションのリスト
        """
        recommendations = []
        
        # 原因に基づく一般的な対応策
        general_actions = {
            'database_connection_limit': [
                {
                    'action': 'Increase connection pool size',
                    'priority': 'HIGH',
                    'implementation': 'Update max_connections in database configuration'
                },
                {
                    'action': 'Implement connection pooling',
                    'priority': 'MEDIUM',
                    'implementation': 'Deploy connection pooling middleware'
                }
            ],
            'memory_leak': [
                {
                    'action': 'Analyze heap dumps',
                    'priority': 'HIGH',
                    'implementation': 'Use memory profiler to identify leaking objects'
                },
                {
                    'action': 'Implement memory monitoring',
                    'priority': 'MEDIUM',
                    'implementation': 'Deploy memory monitoring agents'
                }
            ],
            'deadlock': [
                {
                    'action': 'Review transaction isolation levels',
                    'priority': 'HIGH',
                    'implementation': 'Adjust database transaction isolation settings'
                },
                {
                    'action': 'Implement deadlock detection',
                    'priority': 'MEDIUM',
                    'implementation': 'Deploy deadlock monitoring solution'
                }
            ]
        }
        
        # 原因特有の推奨アクション
        if root_cause['primary_cause'] in general_actions:
            recommendations.extend(general_actions[root_cause['primary_cause']])
        
        # コンポーネント特有の推奨アクション
        for component in root_cause['affected_components']:
            component_actions = self._get_component_specific_actions(component)
            recommendations.extend(component_actions)
        
        # 優先度でソート
        recommendations.sort(key=lambda x: {'HIGH