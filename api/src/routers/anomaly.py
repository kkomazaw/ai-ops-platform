# api/src/routers/ecommerce.py
@router.post("/analyze/ecommerce")
async def analyze_ecommerce_metrics(
    metrics: schemas.ECommerceMetrics,
    detector: ECommerceDetector = Depends(get_detector)
):
    results = await detector.detect_anomalies(metrics.dict())
    return results