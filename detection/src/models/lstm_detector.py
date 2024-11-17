import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import logging
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMDetector:
    def __init__(self, 
                 sequence_length: int = 60,
                 feature_dim: int = 6,
                 lstm_units: int = 64):
        """
        LSTMベースの異常検知モデル

        Args:
            sequence_length (int): 入力シーケンスの長さ
            feature_dim (int): 特徴量の次元数
            lstm_units (int): LSTMレイヤーのユニット数
        """
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.lstm_units = lstm_units
        self.model = self._build_model()
        self.threshold = None

    def _build_model(self) -> Sequential:
        """LSTMモデルの構築"""
        model = Sequential([
            LSTM(self.lstm_units, 
                 input_shape=(self.sequence_length, self.feature_dim),
                 return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(self.feature_dim)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, 
             train_data: np.ndarray,
             validation_split: float = 0.2,
             epochs: int = 50,
             batch_size: int = 32) -> None:
        """
        モデルの学習

        Args:
            train_data (np.ndarray): 学習データ
            validation_split (float): 検証データの割合
            epochs (int): エポック数
            batch_size (int): バッチサイズ
        """
        try:
            history = self.model.fit(
                train_data,
                train_data,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
            
            # 再構成誤差の分布から閾値を設定
            predictions = self.model.predict(train_data)
            reconstruction_errors = np.mean(np.abs(train_data - predictions), axis=1)
            self.threshold = np.percentile(reconstruction_errors, 95)
            
            logger.info(f"Model training completed. Threshold set to: {self.threshold}")
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise

    def detect_anomalies(self, data: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        異常検知の実行

        Args:
            data (np.ndarray): 入力データ

        Returns:
            Tuple[np.ndarray, List[Dict]]: 異常スコアと異常検知結果
        """
        try:
            # 予測と再構成誤差の計算
            predictions = self.model.predict(data)
            reconstruction_errors = np.mean(np.abs(data - predictions), axis=1)
            
            # 異常判定
            anomalies = []
            for i, error in enumerate(reconstruction_errors):
                is_anomaly = error > self.threshold
                anomalies.append({
                    'timestamp': i,  # 実際のタイムスタンプに置き換える
                    'error_score': float(error),
                    'is_anomaly': bool(is_anomaly),
                    'severity': self._calculate_severity(error)
                })
            
            return reconstruction_errors, anomalies
            
        except Exception as e:
            logger.error(f"Error during anomaly detection: {e}")
            raise

    def _calculate_severity(self, error: float) -> str:
        """
        異常の重要度を計算

        Args:
            error (float): 再構成誤差

        Returns:
            str: 重要度レベル（LOW/MEDIUM/HIGH/CRITICAL）
        """
        if error > self.threshold * 3:
            return 'CRITICAL'
        elif error > self.threshold * 2:
            return 'HIGH'
        elif error > self.threshold * 1.5:
            return 'MEDIUM'
        else:
            return 'LOW'

    def save_model(self, path: str) -> None:
        """モデルの保存"""
        try:
            self.model.save(path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, path: str) -> None:
        """モデルの読み込み"""
        try:
            self.model = tf.keras.models.load_model(path)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
