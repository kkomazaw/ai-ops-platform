@router.post("/incidents/{incident_id}/analyze")
async def analyze_incident(
    incident_id: str,
    analysis_service: RootCauseAnalysisService = Depends(get_analysis_service)
):
    incident = await get_incident(incident_id)
    metrics = await get_incident_metrics(incident_id)
    logs = await get_incident_logs(incident_id)
    
    analysis_result = await analysis_service.analyze_incident(
        incident_id=incident_id,
        metrics=metrics,
        logs=logs
    )
    
    return analysis_result