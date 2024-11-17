from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from transformers import pipeline
import torch
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Incident:
    """インシデント情報"""
    id: str
    title: str
    description: str
    severity: str
    service: str
    metrics: Dict[str, float]
    timestamp: datetime

@dataclass
class Solution:
    """解決策情報"""
    id: str
    title: str
    description: str
    steps: List[str]
    prerequisites: List[str]
    estimated_time: int  # 分
    risk_level: str
    success_rate: float
    automation_possible: bool

@dataclass
class Recommendation:
    """推奨内容"""
    solution: Solution
    confidence: float
    reasoning: str
    priority: int
    additional_context: Dict
    related_incidents: List[str]

class AIRecommender:
    def __init__(self, model_path: Optional[str] = None):
        """
        AI駆動の解決策推奨エンジン

        Args:
            model_path (Optional[str]): 事前学習済みモデルのパス
        """
        self.classifier = RandomForestClassifier()
        self.solution_database = self._initialize_solutions()
        self.historical_data = []
        self.nlp_pipeline = pipeline("text-classification")
        self.success_metrics = defaultdict(list)

        # モデルのロード
        if model_path:
            self._load_model(model_path)

    def _initialize_solutions(self) -> Dict[str, Solution]:
        """解決策データベースの初期化"""
        return {
            "DB_CONN_001": Solution(
                id="DB_CONN_001",
                title="Database Connection Pool Optimization",
                description="Optimize database connection pool settings to handle high load",
                steps=[
                    "Check current connection pool metrics",
                    "Calculate optimal pool size based on traffic",
                    "Gradually increase max_connections",
                    "Monitor for connection timeouts",
                    "Update HikariCP configuration"
                ],
                prerequisites=["Database metrics access", "Admin permissions"],
                estimated_time=30,
                risk_level="medium",
                success_rate=0.85,
                automation_possible=True
            ),
            "MEM_LEAK_001": Solution(
                id="MEM_LEAK_001",
                title="Memory Leak Investigation and Resolution",
                description="Investigate and fix memory leaks in Java applications",
                steps=[
                    "Generate heap dump",
                    "Analyze with Memory Analyzer",
                    "Identify memory leak sources",
                    "Apply fixes to resource cleanup",
                    "Verify memory usage patterns"
                ],
                prerequisites=["JVM access", "Heap dump permissions"],
                estimated_time=120,
                risk_level="high",
                success_rate=0.75,
                automation_possible=False
            ),
            "API_PERF_001": Solution(
                id="API_PERF_001",
                title="API Performance Optimization",
                description="Optimize API endpoint performance and response times",
                steps=[
                    "Profile API endpoints",
                    "Identify bottlenecks",
                    "Implement caching",
                    "Optimize database queries",
                    "Add response compression"
                ],
                prerequisites=["API access", "Performance metrics"],
                estimated_time=60,
                risk_level="low",
                success_rate=0.9,
                automation_possible=True
            )
        }

    def recommend_solutions(self, 
                          incident: Incident,
                          max_recommendations: int = 3) -> List[Recommendation]:
        """
        インシデントに対する解決策の推奨

        Args:
            incident (Incident): インシデント情報
            max_recommendations (int): 最大推奨数

        Returns:
            List[Recommendation]: 推奨解決策のリスト
        """
        try:
            # 特徴量の抽出
            features = self._extract_features(incident)
            
            # 類似インシデントの検索
            similar_incidents = self._find_similar_incidents(incident)
            
            # 解決策候補の評価
            candidates = self._evaluate_solutions(
                incident,
                features,
                similar_incidents
            )
            
            # コンテキストに基づく優先順位付け
            prioritized = self._prioritize_solutions(
                candidates,
                incident
            )
            
            # 推奨内容の生成
            recommendations = []
            for i, (solution_id, score) in enumerate(prioritized[:max_recommendations]):
                solution = self.solution_database[solution_id]
                
                recommendation = Recommendation(
                    solution=solution,
                    confidence=score,
                    reasoning=self._generate_reasoning(
                        solution,
                        incident,
                        score
                    ),
                    priority=i + 1,
                    additional_context=self._get_additional_context(
                        solution,
                        incident
                    ),
                    related_incidents=[inc.id for inc in similar_incidents]
                )
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            raise

    def _extract_features(self, incident: Incident) -> np.ndarray:
        """特徴量の抽出"""
        features = []
        
        # メトリクス特徴量
        for metric_name in ['cpu_usage', 'memory_usage', 'error_rate', 'latency']:
            features.append(incident.metrics.get(metric_name, 0))
        
        # 重要度の数値化
        severity_map = {
            'low': 1,
            'medium': 2,
            'high': 3,
            'critical': 4
        }
        features.append(severity_map.get(incident.severity.lower(), 0))
        
        # テキスト特徴量
        text_features = self.nlp_pipeline(
            incident.description,
            return_tensors=True
        )
        features.extend(text_features[0]['hidden_states'][-1][0].numpy())
        
        return np.array(features)

    def _find_similar_incidents(self, 
                              incident: Incident,
                              threshold: float = 0.7) -> List[Incident]:
        """類似インシデントの検索"""
        similar_incidents = []
        incident_features = self._extract_features(incident)
        
        for hist_incident in self.historical_data:
            hist_features = self._extract_features(hist_incident)
            similarity = self._calculate_similarity(
                incident_features,
                hist_features
            )
            
            if similarity >= threshold:
                similar_incidents.append(hist_incident)
        
        return sorted(
            similar_incidents,
            key=lambda x: self._calculate_similarity(
                incident_features,
                self._extract_features(x)
            ),
            reverse=True
        )

    def _evaluate_solutions(self,
                          incident: Incident,
                          features: np.ndarray,
                          similar_incidents: List[Incident]) -> Dict[str, float]:
        """解決策候補の評価"""
        solution_scores = defaultdict(float)
        
        # 特徴量に基づく評価
        for solution_id, solution in self.solution_database.items():
            base_score = self._calculate_base_score(solution, incident)
            
            # 成功率の考慮
            base_score *= solution.success_rate
            
            # リスクレベルの調整
            risk_multiplier = {
                'low': 1.0,
                'medium': 0.8,
                'high': 0.6
            }.get(solution.risk_level.lower(), 0.5)
            
            adjusted_score = base_score * risk_multiplier
            solution_scores[solution_id] = adjusted_score
        
        # 類似インシデントからの知見
        for similar in similar_incidents:
            if similar.id in self.success_metrics:
                for solution_id, success in self.success_metrics[similar.id]:
                    if success:
                        solution_scores[solution_id] *= 1.2  # 成功事例の重み付け
        
        return solution_scores

    def _prioritize_solutions(self,
                            candidates: Dict[str, float],
                            incident: Incident) -> List[Tuple[str, float]]:
        """解決策の優先順位付け"""
        prioritized = []
        
        for solution_id, score in candidates.items():
            solution = self.solution_database[solution_id]
            
            # 重要度に基づく調整
            if incident.severity.lower() == 'critical':
                if solution.estimated_time <= 30:  # 30分以内
                    score *= 1.5
            
            # 自動化可能性の考慮
            if solution.automation_possible:
                score *= 1.2
            
            # 前提条件の確認
            prerequisites_met = self._check_prerequisites(
                solution.prerequisites,
                incident
            )
            if not prerequisites_met:
                score *= 0.5
            
            prioritized.append((solution_id, score))
        
        return sorted(prioritized, key=lambda x: x[1], reverse=True)

    def _generate_reasoning(self,
                          solution: Solution,
                          incident: Incident,
                          score: float) -> str:
        """推奨理由の生成"""
        reasons = []
        
        # 成功率に基づく理由
        if solution.success_rate >= 0.8:
            reasons.append(
                f"High success rate ({solution.success_rate*100:.0f}%) "
                "in similar situations"
            )
        
        # 重要度に基づく理由
        if incident.severity.lower() == 'critical':
            if solution.estimated_time <= 30:
                reasons.append(
                    "Quick solution suitable for critical incident"
                )
        
        # 自動化可能性
        if solution.automation_possible:
            reasons.append("Can be automated for faster resolution")
        
        # スコアに基づく確信度
        confidence_level = "high" if score >= 0.8 else "medium" if score >= 0.5 else "low"
        reasons.append(f"Solution confidence level: {confidence_level}")
        
        return " | ".join(reasons)

    def _get_additional_context(self,
                              solution: Solution,
                              incident: Incident) -> Dict:
        """追加コンテキストの取得"""
        return {
            'estimated_time': solution.estimated_time,
            'risk_level': solution.risk_level,
            'automation_possible': solution.automation_possible,
            'prerequisites': solution.prerequisites,
            'related_metrics': {
                k: v for k, v in incident.metrics.items()
                if k in ['cpu_usage', 'memory_usage', 'error_rate']
            }
        }

    def update_success_metrics(self,
                             incident_id: str,
                             solution_id: str,
                             success: bool) -> None:
        """解決策の成功メトリクスの更新"""
        self.success_metrics[incident_id].append((solution_id, success))
        
        # 成功率の更新
        if solution_id in self.solution_database:
            solution = self.solution_database[solution_id]
            success_count = sum(1 for s_id, s in self.success_metrics[incident_id]
                              if s_id == solution_id and s)
            total_count = sum(1 for s_id, _ in self.success_metrics[incident_id]
                            if s_id == solution_id)
            
            solution.success_rate = success_count / total_count

    def add_solution(self, solution: Solution) -> None:
        """新しい解決策の追加"""
        if solution.id in self.solution_database:
            raise ValueError(f"Solution {solution.id} already exists")
        
        self.solution_database[solution.id] = solution
        logger.info(f"Added new solution: {solution.id}")

    def _calculate_similarity(self,
                            features1: np.ndarray,
                            features2: np.ndarray) -> float:
        """特徴量ベクトル間の類似度計算"""
        return float(np.dot(features1, features2) / 
                    (np.linalg.norm(features1) * np.linalg.norm(features2)))

    def _calculate_base_score(self,
                            solution: Solution,
                            incident: Incident) -> float:
        """基本スコアの計算"""
        score = 0.0
        
        # 重要度の一致
        if incident.severity.lower() == solution.risk_level.lower():
            score += 0.3
        
        # 推定時間の評価
        if incident.severity.lower() == 'critical':
            score += (1.0 - min(solution.estimated_time / 120, 1.0)) * 0.3
        
        # 成功率の考慮
        score += solution.success_rate * 0.4
        
        return score

    def _check_prerequisites(self,
                           prerequisites: List[str],
                           incident: Incident) -> bool:
        """前提条件の確認"""
        # 実際の環境では、より詳細なチェックが必要
        return True

    def save_model(self, filepath: str) -> None:
        """モデルの保存"""
        try:
            model_data = {
                'solutions': self.solution_database,
                'success_metrics': self.success_metrics,
                'classifier': self.classifier
            }
            
            with open(filepath, 'wb') as f:
                torch.save(model_data, f)
                
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def _load_model(self, filepath: str) -> None:
        """モデルのロード"""
        try:
            with open(filepath, 'rb') as f:
                model_data = torch.load(f)
            
            self.solution_database = model_data['solutions']
            self.success_metrics = model_data['success_metrics']
            self.classifier = model_data['classifier']
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
