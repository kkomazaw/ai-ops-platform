from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from ..db import database, models, schemas
from ..utils.security import get_current_user
from ..services import incident_service

router = APIRouter()

@router.get("/", response_model=List[schemas.Incident])
async def get_incidents(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    severity: Optional[str] = None,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """インシデントの一覧を取得"""
    incidents = incident_service.get_incidents(
        db,
        skip=skip,
        limit=limit,
        status=status,
        severity=severity,
        from_date=from_date,
        to_date=to_date
    )
    return incidents

@router.post("/", response_model=schemas.Incident)
async def create_incident(
    incident: schemas.IncidentCreate,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """新規インシデントの作成"""
    return incident_service.create_incident(db, incident, current_user)

@router.get("/{incident_id}", response_model=schemas.IncidentDetail)
async def get_incident(
    incident_id: str,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """インシデントの詳細を取得"""
    incident = incident_service.get_incident(db, incident_id)
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    return incident

@router.put("/{incident_id}", response_model=schemas.Incident)
async def update_incident(
    incident_id: str,
    incident_update: schemas.IncidentUpdate,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """インシデントの更新"""
    incident = incident_service.update_incident(
        db,
        incident_id,
        incident_update,
        current_user
    )
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    return incident

@router.post("/{incident_id}/events", response_model=schemas.IncidentEvent)
async def add_incident_event(
    incident_id: str,
    event: schemas.IncidentEventCreate,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """インシデントイベントの追加"""
    return incident_service.add_incident_event(
        db,
        incident_id,
        event,
        current_user
    )

@router.get("/{incident_id}/events", response_model=List[schemas.IncidentEvent])
async def get_incident_events(
    incident_id: str,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """インシデントイベントの取得"""
    return incident_service.get_incident_events(
        db,
        incident_id,
        skip=skip,
        limit=limit
    )

@router.post("/{incident_id}/acknowledge")
async def acknowledge_incident(
    incident_id: str,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """インシデントの確認"""
    incident = incident_service.acknowledge_incident(
        db,
        incident_id,
        current_user
    )
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    return {"status": "acknowledged"}

@router.post("/{incident_id}/resolve")
async def resolve_incident(
    incident_id: str,
    resolution: schemas.IncidentResolution,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """インシデントの解決"""
    incident = incident_service.resolve_incident(
        db,
        incident_id,
        resolution,
        current_user
    )
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    return {"status": "resolved"}

@router.get("/{incident_id}/metrics", response_model=List[schemas.MetricData])
async def get_incident_metrics(
    incident_id: str,
    metric_name: Optional[str] = None,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """インシデント関連のメトリクスを取得"""
    return incident_service.get_incident_metrics(
        db,
        incident_id,
        metric_name=metric_name,
        from_date=from_date,
        to_date=to_date
    )

@router.get("/{incident_id}/analysis", response_model=schemas.AnalysisResult)
async def get_incident_analysis(
    incident_id: str,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """インシデントの分析結果を取得"""
    analysis = incident_service.get_incident_analysis(db, incident_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis

@router.get("/{incident_id}/remediation", response_model=List[schemas.RemediationAction])
async def get_incident_remediation_actions(
    incident_id: str,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """インシデントの修復アクションを取得"""
    return incident_service.get_remediation_actions(db, incident_id)
