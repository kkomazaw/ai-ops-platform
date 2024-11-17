from .models import RootCauseAnalyzer, BertAnalyzer, RandomForestAnalyzer

class RootCauseAnalysisService:
    def __init__(self):
        self.analyzer = RootCauseAnalyzer()
        self.bert_analyzer = BertAnalyzer()
        self.rf_analyzer = RandomForestAnalyzer()
    
    async def analyze_incident(self, incident_id: str, metrics: Dict, logs: List[str]) -> Dict:
        # メトリクスとログの分析
        metrics_analysis = self.rf_analyzer.analyze_metrics(metrics)
        log_analysis = self.bert_analyzer.analyze_logs(logs)
        
        # 総合的な原因分析
        root_cause = self.analyzer.analyze_root_cause(
            anomaly_data=metrics_analysis,
            system_logs=log_analysis,
            metrics_history=metrics
        )
        
        return root_cause