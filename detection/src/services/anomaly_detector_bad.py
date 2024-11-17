# detection/src/models/detectors/ecommerce_detector.py
class ECommerceDetector:
    def __init__(self):
        self.lstm_detector = LSTMDetector()  # 既存のLSTM検知器
        self.statistical_detector = StatisticalDetector()  # 既存の統計的検知器
        
    async def detect_anomalies(self, metrics: Dict[str, float]) -> Dict:
        # 異常検知の実行
        lstm_results = self.lstm_detector.detect_anomalies(metrics)
        stat_results = self.statistical_detector.detect_anomalies(metrics)
        
        return {
            'lstm_anomalies': lstm_results,
            'statistical_anomalies': stat_results,
            'combined_score': self._combine_scores(lstm_results, stat_results)
        }