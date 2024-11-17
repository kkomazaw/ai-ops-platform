import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from collections import deque
import logging
import json

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnomalyDetector:
    def __init__(self, window_size=60, threshold_std=3):
        """
        異常検知システムの初期化
        
        Parameters:
        window_size (int): 移動窓のサイズ（デフォルト: 60秒）
        threshold_std (float): 異常判定の標準偏差閾値（デフォルト: 3シグマ）
        """
        self.window_size = window_size
        self.threshold_std = threshold_std
        self.scaler = StandardScaler()
        self.lstm_model = self._build_lstm_model()
        self.metric_windows = {
            'active_users': deque(maxlen=window_size),
            'db_connections': deque(maxlen=window_size),
            'transaction_time': deque(maxlen=window_size),
            'error_rate': deque(maxlen=window_size),
            'cpu_usage': deque(maxlen=window_size),
            'memory_usage': deque(maxlen=window_size)
        }
        self.thresholds = self._load_thresholds()

    def _load_thresholds(self):
        """閾値設定の読み込み"""
        return {
            'active_users': {'warning': 500, 'critical': 800},
            'db_connections': {'warning': 80, 'critical': 90},
            'transaction_time': {'warning': 1000, 'critical': 2000},
            'error_rate': {'warning': 0.01, 'critical': 0.05},
            'cpu_usage': {'warning': 70, 'critical': 90},
            'memory_usage': {'warning': 80, 'critical': 95}
        }

    def _build_lstm_model(self):
        """LSTMモデルの構築"""
        model = Sequential([
            LSTM(64, input_shape=(self.window_size, 6), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(6)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def update_metrics(self, metrics):
        """
        新しいメトリクスデータの追加と異常検知
        
        Parameters:
        metrics (dict): 各メトリクスの最新値
        
        Returns:
        dict: 異常検知結果
        """
        # メトリクスの更新
        for metric_name, value in metrics.items():
            if metric_name in self.metric_windows:
                self.metric_windows[metric_name].append(value)

        # 異常検知の実行
        anomalies = self._detect_anomalies()
        
        return anomalies

    def _detect_anomalies(self):
        """
        複数の手法を組み合わせた異常検知の実行
        
        Returns:
        dict: 検出された異常のリスト
        """
        anomalies = {
            'statistical': self._statistical_analysis(),
            'threshold': self._threshold_analysis(),
            'lstm': self._lstm_analysis()
        }

        # 総合的な異常スコアの計算
        return self._combine_anomaly_results(anomalies)

    def _statistical_analysis(self):
        """
        統計的手法による異常検知
        
        Returns:
        dict: 各メトリクスの統計的異常値
        """
        statistical_anomalies = {}
        
        for metric_name, window in self.metric_windows.items():
            if len(window) >= self.window_size:
                values = np.array(list(window))
                z_scores = stats.zscore(values)
                
                # 最新値のZ-scoreが閾値を超えているか確認
                current_z_score = z_scores[-1]
                is_anomaly = abs(current_z_score) > self.threshold_std
                
                statistical_anomalies[metric_name] = {
                    'is_anomaly': is_anomaly,
                    'z_score': current_z_score,
                    'current_value': values[-1]
                }

        return statistical_anomalies

    def _threshold_analysis(self):
        """
        閾値ベースの異常検知
        
        Returns:
        dict: 閾値超過の検出結果
        """
        threshold_anomalies = {}
        
        for metric_name, window in self.metric_windows.items():
            if len(window) > 0:
                current_value = window[-1]
                thresholds = self.thresholds[metric_name]
                
                threshold_anomalies[metric_name] = {
                    'is_warning': current_value >= thresholds['warning'],
                    'is_critical': current_value >= thresholds['critical'],
                    'current_value': current_value
                }

        return threshold_anomalies

    def _lstm_analysis(self):
        """
        LSTMによる予測ベースの異常検知
        
        Returns:
        dict: LSTM予測との乖離による異常値
        """
        if all(len(window) >= self.window_size for window in self.metric_windows.values()):
            # 入力データの準備
            input_data = np.array([[list(window)] for window in self.metric_windows.values()])
            input_data = input_data.reshape(1, self.window_size, 6)
            
            # 予測の実行
            prediction = self.lstm_model.predict(input_data)
            
            # 現在値との差分を計算
            current_values = np.array([[window[-1] for window in self.metric_windows.values()]])
            deviation = np.abs(prediction - current_values)
            
            return {
                metric_name: {
                    'predicted': float(pred),
                    'actual': float(actual),
                    'deviation': float(dev)
                }
                for metric_name, pred, actual, dev in zip(
                    self.metric_windows.keys(),
                    prediction[0],
                    current_values[0],
                    deviation[0]
                )
            }
        
        return {}

    def _combine_anomaly_results(self, anomalies):
        """
        各種異常検知結果の組み合わせと重み付け
        
        Parameters:
        anomalies (dict): 各手法による異常検知結果
        
        Returns:
        dict: 統合された異常検知結果
        """
        combined_results = {}
        
        for metric_name in self.metric_windows.keys():
            # 各検知手法の結果を統合
            statistical_score = 1 if anomalies['statistical'].get(metric_name, {}).get('is_anomaly', False) else 0
            threshold_score = (
                2 if anomalies['threshold'].get(metric_name, {}).get('is_critical', False)
                else 1 if anomalies['threshold'].get(metric_name, {}).get('is_warning', False)
                else 0
            )
            
            lstm_deviation = anomalies['lstm'].get(metric_name, {}).get('deviation', 0)
            lstm_score = 1 if lstm_deviation > np.mean(list(self.metric_windows[metric_name])) else 0
            
            # 総合スコアの計算（重み付け）
            total_score = (
                statistical_score * 0.3 +
                threshold_score * 0.4 +
                lstm_score * 0.3
            )
            
            # 異常度の判定
            severity = (
                'CRITICAL' if total_score >= 1.5
                else 'WARNING' if total_score >= 0.8
                else 'NORMAL'
            )
            
            combined_results[metric_name] = {
                'severity': severity,
                'score': total_score,
                'details': {
                    'statistical': anomalies['statistical'].get(metric_name, {}),
                    'threshold': anomalies['threshold'].get(metric_name, {}),
                    'lstm': anomalies['lstm'].get(metric_name, {})
                }
            }

        return combined_results

    def get_alert_message(self, anomaly_results):
        """
        異常検知結果からアラートメッセージを生成
        
        Parameters:
        anomaly_results (dict): 異常検知結果
        
        Returns:
        str: アラートメッセージ
        """
        messages = []
        
        for metric_name, result in anomaly_results.items():
            if result['severity'] != 'NORMAL':
                details = result['details']
                current_value = details['threshold'].get('current_value', 'N/A')
                
                message = (
                    f"[{result['severity']}] {metric_name}: "
                    f"Current value: {current_value}, "
                    f"Anomaly score: {result['score']:.2f}"
                )
                messages.append(message)
        
        return '\n'.join(messages) if messages else "No anomalies detected."

class AnomalyDetectionSystem:
    def __init__(self):
        """異常検知システムの初期化"""
        self.detector = AnomalyDetector()
        
    def process_metrics(self, metrics_data):
        """
        メトリクスデータの処理と異常検知の実行
        
        Parameters:
        metrics_data (dict): Prometheusから取得したメトリクスデータ
        
        Returns:
        tuple: (異常検知結果, アラートメッセージ)
        """
        try:
            # メトリクスの更新と異常検知
            anomaly_results = self.detector.update_metrics(metrics_data)
            
            # アラートメッセージの生成
            alert_message = self.detector.get_alert_message(anomaly_results)
            
            # 重大な異常がある場合はログに記録
            if any(result['severity'] == 'CRITICAL' for result in anomaly_results.values()):
                logger.warning(f"Critical anomalies detected:\n{alert_message}")
            
            return anomaly_results, alert_message
            
        except Exception as e:
            logger.error(f"Error processing metrics: {e}")
            raise

if __name__ == "__main__":
    # 異常検知システムのテスト
    system = AnomalyDetectionSystem()
    
    # テストデータの生成
    test_metrics = {
        'active_users': 850,  # 危険閾値超過
        'db_connections': 85,  # 警告閾値超過
        'transaction_time': 1500,  # 警告閾値超過
        'error_rate': 0.06,  # 危険閾値超過
        'cpu_usage': 95,  # 危険閾値超過
        'memory_usage': 88  # 警告閾値超過
    }
    
    # 異常検知の実行
    results, alert = system.process_metrics(test_metrics)
    
    # 結果の出力
    print("Anomaly Detection Results:")
    print(json.dumps(results, indent=2))
    print("\nAlert Message:")
    print(alert)