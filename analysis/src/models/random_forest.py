import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional
import joblib
import logging
from datetime import datetime
from dataclasses import dataclass
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MetricData:
    """メトリクスデータの構造"""
    timestamp: datetime
    name: str
    value: float
    labels: Dict[str, str]

@dataclass
class AnalysisResult:
    """分析結果の構造"""
    metric_name: str
    prediction: str
    probability: float
    feature_importance: Dict[str, float]
    related_metrics: List[str]
    threshold_violations: Dict[str, float]
    trend_analysis: Dict[str, str]

class RandomForestAnalyzer:
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 random_state: int = 42):
        """
        Random Forest based メトリクス分析モデル

        Args:
            n_estimators (int): 決定木の数
            max_depth (Optional[int]): 決定木の最大深さ
            random_state (int): 乱数シード
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.threshold_configs = self._initialize_thresholds()
        self.correlation_threshold = 0.7

    def _initialize_thresholds(self) -> Dict:
        """閾値設定の初期化"""
        return {
            'cpu_usage': {
                'warning': 70,
                'critical': 90,
                'trend_window': 5
            },
            'memory_usage': {
                'warning': 80,
                'critical': 95,
                'trend_window': 5
            },
            'response_time': {
                'warning': 1000,  # milliseconds
                'critical': 2000,
                'trend_window': 3
            },
            'error_rate': {
                'warning': 0.01,  # 1%
                'critical': 0.05,  # 5%
                'trend_window': 3
            },
            'request_rate': {
                'warning': 1000,
                'critical': 2000,
                'trend_window': 5
            }
        }

    def prepare_features(self, 
                        metrics: List[MetricData]) -> Tuple[np.ndarray, List[str]]:
        """
        特徴量の準備

        Args:
            metrics (List[MetricData]): メトリクスデータ

        Returns:
            Tuple[np.ndarray, List[str]]: 特徴量配列と特徴量名のリスト
        """
        # メトリクスデータをDataFrameに変換
        df = pd.DataFrame([
            {
                'timestamp': m.timestamp,
                'name': m.name,
                'value': m.value,
                **m.labels
            }
            for m in metrics
        ])

        # 時系列特徴量の作成
        features = []
        feature_names = []

        for metric_name in df['name'].unique():
            metric_data = df[df['name'] == metric_name]['value'].values
            
            # 基本統計量
            features.extend([
                np.mean(metric_data),
                np.std(metric_data),
                np.min(metric_data),
                np.max(metric_data)
            ])
            feature_names.extend([
                f'{metric_name}_mean',
                f'{metric_name}_std',
                f'{metric_name}_min',
                f'{metric_name}_max'
            ])

            # トレンド特徴量
            if len(metric_data) > 1:
                trend = np.polyfit(range(len(metric_data)), metric_data, 1)[0]
                features.append(trend)
                feature_names.append(f'{metric_name}_trend')

            # 変化率
            if len(metric_data) > 1:
                change_rate = (metric_data[-1] - metric_data[0]) / metric_data[0]
                features.append(change_rate)
                feature_names.append(f'{metric_name}_change_rate')

        self.feature_names = feature_names
        features_array = np.array(features).reshape(1, -1)
        return self.scaler.fit_transform(features_array), feature_names

    def train(self, 
             training_data: List[MetricData],
             labels: List[str]) -> None:
        """
        モデルの学習

        Args:
            training_data (List[MetricData]): 学習用メトリクスデータ
            labels (List[str]): 正解ラベル
        """
        try:
            # 特徴量の準備
            X, feature_names = self.prepare_features(training_data)
            y = np.array(labels)

            # モデルの学習
            self.model.fit(X, y)
            logger.info("Model training completed successfully")

            # 特徴量の重要度を記録
            self.feature_importance = dict(zip(
                feature_names,
                self.model.feature_importances_
            ))

        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise

    def analyze_metrics(self, metrics: List[MetricData]) -> AnalysisResult:
        """
        メトリクスの分析を実行

        Args:
            metrics (List[MetricData]): 分析対象のメトリクスデータ

        Returns:
            AnalysisResult: 分析結果
        """
        try:
            # 特徴量の抽出
            X, _ = self.prepare_features(metrics)

            # 予測の実行
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            max_prob = max(probabilities)

            # 関連メトリクスの特定
            related_metrics = self._identify_related_metrics(metrics)

            # 閾値違反の確認
            threshold_violations = self._check_thresholds(metrics)

            # トレンド分析
            trend_analysis = self._analyze_trends(metrics)

            # 結果の生成
            result = AnalysisResult(
                metric_name=metrics[0].name,
                prediction=prediction,
                probability=float(max_prob),
                feature_importance=self.feature_importance,
                related_metrics=related_metrics,
                threshold_violations=threshold_violations,
                trend_analysis=trend_analysis
            )

            return result

        except Exception as e:
            logger.error(f"Error during metrics analysis: {e}")
            raise

    def _identify_related_metrics(self, metrics: List[MetricData]) -> List[str]:
        """
        関連するメトリクスの特定

        Args:
            metrics (List[MetricData]): メトリクスデータ

        Returns:
            List[str]: 関連メトリクス名のリスト
        """
        df = pd.DataFrame([
            {
                'timestamp': m.timestamp,
                'name': m.name,
                'value': m.value
            }
            for m in metrics
        ])

        related_metrics = []
        pivot_df = df.pivot(
            index='timestamp',
            columns='name',
            values='value'
        )

        # 相関分析
        corr_matrix = pivot_df.corr()
        for col in corr_matrix.columns:
            correlations = corr_matrix[col].abs()
            related = correlations[
                (correlations > self.correlation_threshold) &
                (correlations.index != col)
            ].index.tolist()
            related_metrics.extend(related)

        return list(set(related_metrics))

    def _check_thresholds(self, metrics: List[MetricData]) -> Dict[str, float]:
        """閾値違反のチェック"""
        violations = {}
        
        for metric in metrics:
            if metric.name in self.threshold_configs:
                config = self.threshold_configs[metric.name]
                
                if metric.value >= config['critical']:
                    violations[metric.name] = metric.value
                elif metric.value >= config['warning']:
                    violations[metric.name] = metric.value

        return violations

    def _analyze_trends(self, metrics: List[MetricData]) -> Dict[str, str]:
        """トレンド分析の実行"""
        trends = {}
        df = pd.DataFrame([
            {
                'timestamp': m.timestamp,
                'name': m.name,
                'value': m.value
            }
            for m in metrics
        ])

        for name, group in df.groupby('name'):
            if len(group) < 2:
                continue

            values = group['value'].values
            trend = np.polyfit(range(len(values)), values, 1)[0]
            
            if trend > 0.1:
                trends[name] = 'increasing'
            elif trend < -0.1:
                trends[name] = 'decreasing'
            else:
                trends[name] = 'stable'

        return trends

    def save_model(self, filepath: str) -> None:
        """モデルの保存"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'feature_importance': self.feature_importance,
                'threshold_configs': self.threshold_configs
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, filepath: str) -> None:
        """モデルの読み込み"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.feature_importance = model_data['feature_importance']
            self.threshold_configs = model_data['threshold_configs']
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def get_feature_importance(self) -> Dict[str, float]:
        """特徴量の重要度を取得"""
        if not hasattr(self, 'feature_importance'):
            raise ValueError("Model has not been trained yet")
        return dict(sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))

    def update_thresholds(self, 
                         metric_name: str,
                         warning: Optional[float] = None,
                         critical: Optional[float] = None) -> None:
        """閾値の更新"""
        if metric_name not in self.threshold_configs:
            raise ValueError(f"Unknown metric: {metric_name}")

        if warning is not None:
            self.threshold_configs[metric_name]['warning'] = warning
        if critical is not None:
            self.threshold_configs[metric_name]['critical'] = critical

        logger.info(f"Updated thresholds for {metric_name}")
