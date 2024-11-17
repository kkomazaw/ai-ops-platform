import torch
from transformers import BertTokenizer, BertModel
from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime
import json
from dataclasses import dataclass
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LogEntry:
    """ログエントリーの構造"""
    timestamp: datetime
    message: str
    level: str
    service: str
    additional_info: Dict

@dataclass
class AnalysisResult:
    """分析結果の構造"""
    category: str
    confidence: float
    related_logs: List[LogEntry]
    pattern_description: str
    severity: str
    suggested_actions: List[str]

class BertLogAnalyzer:
    def __init__(self, 
                 model_name: str = 'bert-base-uncased',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 batch_size: int = 32,
                 max_length: int = 128):
        """
        BERTを使用したログ分析モデル

        Args:
            model_name (str): BERTモデル名
            device (str): 実行デバイス（'cuda' or 'cpu'）
            batch_size (int): バッチサイズ
            max_length (int): 最大シーケンス長
        """
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        
        # BERTモデルとトークナイザーの初期化
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name)
            self.model = self.model.to(device)
            self.model.eval()
            
            logger.info(f"Initialized BERT model on {device}")
            
        except Exception as e:
            logger.error(f"Error initializing BERT model: {e}")
            raise

        # 既知のパターンの初期化
        self.known_patterns = self._initialize_patterns()

    def _initialize_patterns(self) -> Dict:
        """既知のエラーパターンの初期化"""
        return {
            'database_connection': {
                'keywords': ['connection refused', 'timeout', 'database error'],
                'severity': 'high',
                'actions': [
                    'Check database connectivity',
                    'Verify database credentials',
                    'Check connection pool settings'
                ]
            },
            'memory_error': {
                'keywords': ['out of memory', 'memory leak', 'heap space'],
                'severity': 'critical',
                'actions': [
                    'Analyze memory usage',
                    'Check for memory leaks',
                    'Consider scaling up resources'
                ]
            },
            'api_error': {
                'keywords': ['api failure', 'status 500', 'gateway timeout'],
                'severity': 'medium',
                'actions': [
                    'Check API endpoints',
                    'Verify API authentication',
                    'Monitor API response times'
                ]
            }
        }

    def analyze_logs(self, logs: List[LogEntry]) -> List[AnalysisResult]:
        """
        ログの分析を実行

        Args:
            logs (List[LogEntry]): 分析対象のログエントリーリスト

        Returns:
            List[AnalysisResult]: 分析結果のリスト
        """
        try:
            # ログメッセージの埋め込みベクトルを取得
            embeddings = self._get_embeddings([log.message for log in logs])
            
            # クラスタリングの実行
            clusters = self._cluster_logs(embeddings)
            
            # パターンマッチングと分析
            results = []
            for cluster_id in np.unique(clusters):
                if cluster_id == -1:  # ノイズクラスタ
                    continue
                    
                cluster_indices = np.where(clusters == cluster_id)[0]
                cluster_logs = [logs[i] for i in cluster_indices]
                cluster_embeddings = embeddings[cluster_indices]
                
                # クラスタの分析
                analysis = self._analyze_cluster(
                    cluster_logs,
                    cluster_embeddings
                )
                results.extend(analysis)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing logs: {e}")
            raise

    def _get_embeddings(self, messages: List[str]) -> np.ndarray:
        """
        BERTを使用してログメッセージの埋め込みベクトルを取得

        Args:
            messages (List[str]): ログメッセージのリスト

        Returns:
            np.ndarray: 埋め込みベクトル
        """
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(messages), self.batch_size):
                batch_messages = messages[i:i + self.batch_size]
                
                # トークン化
                encoded = self.tokenizer(
                    batch_messages,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # デバイスに転送
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # BERT による特徴抽出
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # [CLS]トークンの出力を使用
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)

    def _cluster_logs(self, embeddings: np.ndarray) -> np.ndarray:
        """
        ログメッセージのクラスタリング

        Args:
            embeddings (np.ndarray): 埋め込みベクトル

        Returns:
            np.ndarray: クラスタラベル
        """
        # DBSCANによるクラスタリング
        clustering = DBSCAN(
            eps=0.5,
            min_samples=3,
            metric='cosine'
        )
        return clustering.fit_predict(embeddings)

    def _analyze_cluster(self,
                        cluster_logs: List[LogEntry],
                        cluster_embeddings: np.ndarray) -> List[AnalysisResult]:
        """
        クラスタの詳細分析

        Args:
            cluster_logs (List[LogEntry]): クラスタ内のログ
            cluster_embeddings (np.ndarray): クラスタの埋め込みベクトル

        Returns:
            List[AnalysisResult]: 分析結果
        """
        results = []
        
        # クラスタの代表的なログを特定
        centroid = np.mean(cluster_embeddings, axis=0)
        similarities = cosine_similarity([centroid], cluster_embeddings)[0]
        representative_idx = np.argmax(similarities)
        representative_log = cluster_logs[representative_idx]
        
        # パターンマッチング
        for pattern_name, pattern_info in self.known_patterns.items():
            if self._match_pattern(cluster_logs, pattern_info):
                result = AnalysisResult(
                    category=pattern_name,
                    confidence=float(np.max(similarities)),
                    related_logs=cluster_logs,
                    pattern_description=self._generate_pattern_description(
                        pattern_name,
                        cluster_logs
                    ),
                    severity=pattern_info['severity'],
                    suggested_actions=pattern_info['actions']
                )
                results.append(result)
        
        # 未知のパターンの分析
        if not results:
            result = self._analyze_unknown_pattern(
                cluster_logs,
                cluster_embeddings
            )
            results.append(result)
        
        return results

    def _match_pattern(self,
                      logs: List[LogEntry],
                      pattern_info: Dict) -> bool:
        """
        ログが既知のパターンにマッチするか確認

        Args:
            logs (List[LogEntry]): 確認対象のログ
            pattern_info (Dict): パターン情報

        Returns:
            bool: マッチしたかどうか
        """
        # キーワードベースのマッチング
        keywords = pattern_info['keywords']
        match_count = 0
        
        for log in logs:
            if any(keyword.lower() in log.message.lower() for keyword in keywords):
                match_count += 1
        
        # 一定以上のログがマッチした場合にパターンとして認識
        return match_count / len(logs) >= 0.5

    def _analyze_unknown_pattern(self,
                               logs: List[LogEntry],
                               embeddings: np.ndarray) -> AnalysisResult:
        """
        未知のパターンの分析

        Args:
            logs (List[LogEntry]): 分析対象のログ
            embeddings (np.ndarray): 埋め込みベクトル

        Returns:
            AnalysisResult: 分析結果
        """
        # 共通の特徴を抽出
        common_words = self._extract_common_words(logs)
        
        # 重要度の評価
        severity = self._evaluate_severity(logs)
        
        return AnalysisResult(
            category='unknown_pattern',
            confidence=0.6,  # 未知パターンの信頼度は低めに設定
            related_logs=logs,
            pattern_description=f"Unknown pattern with common terms: {', '.join(common_words)}",
            severity=severity,
            suggested_actions=[
                'Monitor this pattern for recurrence',
                'Review logs in detail',
                'Consider adding to known patterns'
            ]
        )

    def _extract_common_words(self, logs: List[LogEntry]) -> List[str]:
        """共通の単語を抽出"""
        word_freq = defaultdict(int)
        total_logs = len(logs)
        
        for log in logs:
            words = set(log.message.lower().split())
            for word in words:
                word_freq[word] += 1
        
        # 一定以上の頻度で出現する単語を抽出
        common_words = [
            word for word, freq in word_freq.items()
            if freq / total_logs >= 0.3 and len(word) > 3
        ]
        
        return common_words[:5]  # 上位5単語を返す

    def _evaluate_severity(self, logs: List[LogEntry]) -> str:
        """ログの重要度を評価"""
        severity_scores = {
            'ERROR': 3,
            'CRITICAL': 4,
            'WARNING': 2,
            'INFO': 1
        }
        
        total_score = sum(
            severity_scores.get(log.level, 1)
            for log in logs
        )
        avg_score = total_score / len(logs)
        
        if avg_score >= 3.5:
            return 'critical'
        elif avg_score >= 2.5:
            return 'high'
        elif avg_score >= 1.5:
            return 'medium'
        else:
            return 'low'

    def _generate_pattern_description(self,
                                   pattern_name: str,
                                   logs: List[LogEntry]) -> str:
        """パターンの説明を生成"""
        pattern_info = self.known_patterns[pattern_name]
        frequency = len(logs)
        time_span = (
            max(log.timestamp for log in logs) -
            min(log.timestamp for log in logs)
        ).total_seconds()
        
        return (
            f"{pattern_name.replace('_', ' ').title()} pattern detected: "
            f"{frequency} occurrences over {time_span:.1f} seconds. "
            f"Severity: {pattern_info['severity']}"
        )

    def save_analysis_results(self,
                            results: List[AnalysisResult],
                            filepath: str):
        """分析結果の保存"""
        serialized_results = []
        for result in results:
            serialized_result = {
                'category': result.category,
                'confidence': result.confidence,
                'pattern_description': result.pattern_description,
                'severity': result.severity,
                'suggested_actions': result.suggested_actions,
                'related_logs': [
                    {
                        'timestamp': log.timestamp.isoformat(),
                        'message': log.message,
                        'level': log.level,
                        'service': log.service,
                        'additional_info': log.additional_info
                    }
                    for log in result.related_logs
                ]
            }
            serialized_results.append(serialized_result)
        
        with open(filepath, 'w') as f:
            json.dump(serialized_results, f, indent=2)
