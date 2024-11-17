from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
import uvicorn
from datetime import datetime, timedelta

from .routers import incidents, metrics, analysis, remediation, services, alerts
from .core import auth, config, logging
from .db import database, models, schemas
from .utils.security import create_access_token, get_current_user

app = FastAPI(
    title="AI Ops Platform API",
    description="AI-driven operations platform API",
    version="1.0.0"
)

# CORSの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ルーターの登録
app.include_router(
    incidents.router,
    prefix="/api/v1/incidents",
    tags=["incidents"]
)
app.include_router(
    metrics.router,
    prefix="/api/v1/metrics",
    tags=["metrics"]
)
app.include_router(
    analysis.router,
    prefix="/api/v1/analysis",
    tags=["analysis"]
)
app.include_router(
    remediation.router,
    prefix="/api/v1/remediation",
    tags=["remediation"]
)
app.include_router(
    services.router,
    prefix="/api/v1/services",
    tags=["services"]
)
app.include_router(
    alerts.router,
    prefix="/api/v1/alerts",
    tags=["alerts"]
)

# 認証エンドポイント
@app.post("/api/v1/auth/token", response_model=schemas.Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(database.get_db)
):
    user = auth.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=config.settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# ヘルスチェックエンドポイント
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow()
    }

# メトリクスエンドポイント
@app.get("/metrics")
async def metrics():
    return {
        "api_requests_total": 100,
        "api_request_duration_seconds": 0.5,
        "api_errors_total": 5
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.settings.HOST,
        port=config.settings.PORT,
        reload=config.settings.DEBUG
    )