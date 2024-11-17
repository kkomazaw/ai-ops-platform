from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from ..db import database, models, schemas
from ..utils.security import get_current_user
from ..services import metrics_service

router = APIRouter()

@router.get("/", response_model=List[schemas.MetricData])
async def get_metrics(
    metric_names: Optional[List[str]] = Query(None),
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    interval: str = "5m",
    aggregation: str = "avg",
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """メトリクスデータの取得"""
    return metrics_service.get_metrics(
        db,
        metric_names=metric_names,
        from_date=from_date,
        to_date=to_date,
        interval=interval,
        aggregation=aggregation
    )

@router.post("/", response_model=schemas.MetricData)
async def create_metric(
    metric: schemas.MetricDataCreate,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """メトリクスデータの登録"""
    return metrics_service.create_metric(db, metric)

@router.get("/summary", response_model=schemas.MetricsSummary)
async def get_metrics_summary(
    metric_names: Optional[List[str]] = Query(None),
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """メトリクスのサマリーを取得"""
    return metrics_service.get_metrics_summary(
        db,
        metric_names=metric_names,
        from_date=from_date,
        to_date=to_date
    )

@router.get("/thresholds", response_model=List[schemas.MetricThreshold])
async def get_metric_thresholds(
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """メトリクスの閾値設定を取得"""
    return metrics_service.get_metric_thresholds(db)

@router.put("/thresholds/{metric_name}", response_model=schemas.MetricThreshold)
async def update_metric_threshold(
    metric_name: str,
    threshold: schemas.MetricThresholdUpdate,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """メトリクスの閾値設定を更新"""
    return metrics_service.update_metric_threshold(
        db,
        metric_name,
        threshold
    )

@router.get("/anomalies", response_model=List[schemas.MetricAnomaly])
async def get_metric_anomalies(
    metric_names: Optional[List[str]] = Query(None),
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    severity: Optional[str] = None,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """メトリクスの異常を取得"""
    return metrics_service.get_metric_anomalies(
        db,
        metric_names=metric_names,
        from_date=from_date,
        to_date=to_date,
        severity=severity
    )

@router.post("/analyze", response_model=schemas.MetricAnalysis)
async def analyze_metrics(
    analysis_request: schemas.MetricAnalysisRequest,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """メトリクスの分析を実行"""
    return metrics_service.analyze_metrics(db, analysis_request)
