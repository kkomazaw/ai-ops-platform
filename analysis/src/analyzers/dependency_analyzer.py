import networkx as nx
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import json
import numpy as np
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ServiceDependency:
    """サービス依存関係の定義"""
    source: str
    target: str
    dependency_type: str  # sync/async
    criticality: str     # high/medium/low
    slo: Optional[float] # Service Level Objective (ms)
    retry_policy: Optional[Dict]

@dataclass
class DependencyMetrics:
    """依存関係のメトリクス"""
    latency: float       # milliseconds
    error_rate: float    # percentage
    traffic: int        # requests per minute
    saturation: float   # percentage of capacity

@dataclass
class ImpactAnalysis:
    """影響分析の結果"""
    affected_services: List[str]
    impact_severity: str
    propagation_path: List[str]
    estimated_users: int
    mitigation_steps: List[str]

class DependencyAnalyzer:
    def __init__(self):
        """依存関係分析エンジンの初期化"""
        self.dependency_graph = nx.DiGraph()
        self.service_metrics = {}
        self.slo_thresholds = self._initialize_slo_thresholds()
        self.critical_paths = set()
        self.cached_analysis = {}

    def _initialize_slo_thresholds(self) -> Dict:
        """SLO閾値の初期化"""
        return {
            'api_gateway': {
                'latency': 200,    # ms
                'error_rate': 0.1,  # 10%
                'saturation': 0.8   # 80%
            },
            'auth_service': {
                'latency': 100,
                'error_rate': 0.05,
                'saturation': 0.7
            },
            'user_service': {
                'latency': 150,
                'error_rate': 0.1,
                'saturation': 0.75
            },
            'payment_service': {
                'latency': 300,
                'error_rate': 0.01,
                'saturation': 0.6
            },
            'notification_service': {
                'latency': 500,
                'error_rate': 0.15,
                'saturation': 0.9
            }
        }

    def add_dependency(self, dependency: ServiceDependency) -> None:
        """
        依存関係の追加

        Args:
            dependency (ServiceDependency): 依存関係の定義
        """
        try:
            self.dependency_graph.add_edge(
                dependency.source,
                dependency.target,
                dependency_type=dependency.dependency_type,
                criticality=dependency.criticality,
                slo=dependency.slo,
                retry_policy=dependency.retry_policy
            )
            
            logger.info(
                f"Added dependency: {dependency.source} -> {dependency.target} "
                f"({dependency.dependency_type})"
            )
            
            # クリティカルパスの更新
            self._update_critical_paths()
            
        except Exception as e:
            logger.error(f"Error adding dependency: {e}")
            raise

    def update_metrics(self, 
                      service: str,
                      metrics: DependencyMetrics) -> None:
        """
        サービスメトリクスの更新

        Args:
            service (str): サービス名
            metrics (DependencyMetrics): メトリクス情報
        """
        self.service_metrics[service] = {
            'latency': metrics.latency,
            'error_rate': metrics.error_rate,
            'traffic': metrics.traffic,
            'saturation': metrics.saturation,
            'updated_at': datetime.now()
        }

    def analyze_impact(self, 
                      failed_service: str,
                      current_metrics: Optional[Dict] = None) -> ImpactAnalysis:
        """
        サービス障害の影響分析

        Args:
            failed_service (str): 障害が発生したサービス
            current_metrics (Optional[Dict]): 現在のメトリクス

        Returns:
            ImpactAnalysis: 影響分析結果
        """
        try:
            # 影響を受けるサービスの特定
            affected_services = self._find_affected_services(failed_service)
            
            # 影響の重大度評価
            severity = self._evaluate_impact_severity(
                failed_service,
                affected_services,
                current_metrics
            )
            
            # 伝播経路の特定
            propagation_path = self._trace_propagation_path(
                failed_service,
                affected_services
            )
            
            # 影響を受けるユーザー数の推定
            estimated_users = self._estimate_affected_users(affected_services)
            
            # 緩和策の提案
            mitigation_steps = self._suggest_mitigation_steps(
                failed_service,
                severity,
                affected_services
            )
            
            return ImpactAnalysis(
                affected_services=list(affected_services),
                impact_severity=severity,
                propagation_path=propagation_path,
                estimated_users=estimated_users,
                mitigation_steps=mitigation_steps
            )
            
        except Exception as e:
            logger.error(f"Error analyzing impact: {e}")
            raise

    def _find_affected_services(self, failed_service: str) -> Set[str]:
        """影響を受けるサービスの特定"""
        affected = set()
        
        # 直接の依存サービスを検索
        for successor in nx.descendants(self.dependency_graph, failed_service):
            affected.add(successor)
            
        # 間接的な影響も考慮
        for service in list(affected):
            affected.update(nx.descendants(self.dependency_graph, service))
            
        return affected

    def _evaluate_impact_severity(self,
                                failed_service: str,
                                affected_services: Set[str],
                                current_metrics: Optional[Dict]) -> str:
        """影響の重大度評価"""
        severity_score = 0
        max_score = 0
        
        # 直接の依存関係の評価
        for service in affected_services:
            edge_data = self.dependency_graph.get_edge_data(
                failed_service,
                service
            )
            if edge_data:
                weight = {
                    'high': 3,
                    'medium': 2,
                    'low': 1
                }.get(edge_data['criticality'], 1)
                
                severity_score += weight
                max_score += 3
                
                # メトリクスベースの評価
                if current_metrics and service in current_metrics:
                    metrics = current_metrics[service]
                    thresholds = self.slo_thresholds.get(service, {})
                    
                    if metrics.get('error_rate', 0) > thresholds.get('error_rate', 1):
                        severity_score += 2
                    if metrics.get('latency', 0) > thresholds.get('latency', float('inf')):
                        severity_score += 1
        
        # スコアの正規化と重大度の判定
        normalized_score = severity_score / max_score if max_score > 0 else 0
        
        if normalized_score >= 0.8:
            return 'CRITICAL'
        elif normalized_score >= 0.5:
            return 'HIGH'
        elif normalized_score >= 0.3:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _trace_propagation_path(self,
                              failed_service: str,
                              affected_services: Set[str]) -> List[str]:
        """障害伝播経路の追跡"""
        paths = []
        for service in affected_services:
            if nx.has_path(self.dependency_graph, failed_service, service):
                path = nx.shortest_path(
                    self.dependency_graph,
                    failed_service,
                    service
                )
                paths.extend(path)
        
        return list(dict.fromkeys(paths))  # 重複を除去

    def _estimate_affected_users(self, affected_services: Set[str]) -> int:
        """影響を受けるユーザー数の推定"""
        total_users = 0
        
        for service in affected_services:
            metrics = self.service_metrics.get(service, {})
            # 1分あたりのトラフィックから概算
            total_users += metrics.get('traffic', 0)
        
        return total_users

    def _suggest_mitigation_steps(self,
                                failed_service: str,
                                severity: str,
                                affected_services: Set[str]) -> List[str]:
        """緩和策の提案"""
        steps = []
        
        # 障害サービスの特性に基づく提案
        edge_data = [
            self.dependency_graph.get_edge_data(failed_service, target)
            for target in self.dependency_graph.successors(failed_service)
        ]
        
        sync_deps = any(
            e and e.get('dependency_type') == 'sync'
            for e in edge_data if e
        )
        
        if sync_deps:
            steps.append(
                f"Implement circuit breaker for {failed_service} "
                "synchronous dependencies"
            )
        
        # 重大度に基づく提案
        if severity in ['CRITICAL', 'HIGH']:
            steps.extend([
                f"Activate failover for {failed_service}",
                "Scale up backup services",
                "Send immediate notification to on-call team"
            ])
        
        # 影響を受けるサービスに基づく提案
        if len(affected_services) > 3:
            steps.append("Consider partial system isolation")
        
        # リトライポリシーの確認
        retry_policies = [
            e.get('retry_policy')
            for e in edge_data if e and e.get('retry_policy')
        ]
        if retry_policies:
            steps.append("Adjust retry policies to prevent cascade")
        
        return steps

    def _update_critical_paths(self) -> None:
        """クリティカルパスの更新"""
        self.critical_paths.clear()
        
        for source in self.dependency_graph.nodes():
            for target in self.dependency_graph.nodes():
                if source != target and nx.has_path(
                    self.dependency_graph,
                    source,
                    target
                ):
                    paths = list(nx.all_simple_paths(
                        self.dependency_graph,
                        source,
                        target
                    ))
                    
                    for path in paths:
                        # パス上の依存関係の重要度を評価
                        critical = all(
                            self.dependency_graph.get_edge_data(
                                path[i],
                                path[i+1]
                            ).get('criticality') == 'high'
                            for i in range(len(path)-1)
                        )
                        
                        if critical:
                            self.critical_paths.add(tuple(path))

    def get_service_health(self) -> Dict[str, str]:
        """サービスの健全性評価"""
        health_status = {}
        
        for service in self.dependency_graph.nodes():
            metrics = self.service_metrics.get(service, {})
            thresholds = self.slo_thresholds.get(service, {})
            
            if not metrics:
                health_status[service] = 'UNKNOWN'
                continue
            
            violations = 0
            if metrics.get('error_rate', 0) > thresholds.get('error_rate', 1):
                violations += 1
            if metrics.get('latency', 0) > thresholds.get('latency', float('inf')):
                violations += 1
            if metrics.get('saturation', 0) > thresholds.get('saturation', 1):
                violations += 1
            
            if violations >= 2:
                health_status[service] = 'UNHEALTHY'
            elif violations == 1:
                health_status[service] = 'DEGRADED'
            else:
                health_status[service] = 'HEALTHY'
        
        return health_status

    def export_topology(self, filepath: str) -> None:
        """依存関係トポロジーのエクスポート"""
        try:
            topology = {
                'nodes': list(self.dependency_graph.nodes()),
                'edges': [
                    {
                        'source': source,
                        'target': target,
                        **self.dependency_graph.get_edge_data(source, target)
                    }
                    for source, target in self.dependency_graph.edges()
                ],
                'critical_paths': list(self.critical_paths),
                'metrics': self.service_metrics
            }
            
            with open(filepath, 'w') as f:
                json.dump(topology, f, indent=2, default=str)
                
            logger.info(f"Topology exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting topology: {e}")
            raise
