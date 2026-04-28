from fastapi import APIRouter
from app.api.detetction.lstm_emb import lstm_emb_router
from app.api.detetction.lstm_attention import lstm_att_router
from app.api.detetction.recursive_hybrid import recursive_router

# ⚡ FIX: remove /api prefix here
api_router = APIRouter(tags=["api"])

# Include model-specific routers
api_router.include_router(lstm_emb_router)
api_router.include_router(lstm_att_router)
api_router.include_router(recursive_router)
