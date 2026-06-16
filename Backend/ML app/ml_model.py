# ML_Apps/ml_model_store.py
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import os
import joblib
from datetime import datetime
from django.conf import settings

FEATURE_VERSION = "v1_0" 

def _default_models_dir() -> Path:
    # <project_root>/ML_models
    return (Path(__file__).resolve().parent.parent / "ML_models").resolve()

def get_models_dir() -> Path:
    base = getattr(settings, "ML_MODELS_DIR", None)
    d = Path(base) if base else _default_models_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d

def _safe_key(key: str) -> str:
    # allow only [A-Za-z0-9_-]; everything else becomes _
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in key)

def model_path(scope_key: str) -> Path:
    models_dir = get_models_dir()
    # sanitize the entire base name so there are no spaces or dots anywhere
    base = f"dropout_rf_{FEATURE_VERSION}__{scope_key}"
    safe_base = _safe_key(base)
    return models_dir / f"{safe_base}.pkl"
    
def try_load(scope_key: str) -> Optional[Dict[str, Any]]:
    p = model_path(scope_key)
    if not p.exists():
        return None
    try:
        obj = joblib.load(p)
        if not isinstance(obj, dict):
            return None
        if obj.get("feature_version") != FEATURE_VERSION:
            return None
        return obj
    except Exception:
        return None

def save_model(scope_key: str, model, scaler, extra_meta: Optional[Dict[str, Any]] = None) -> str:
    p = model_path(scope_key)
    payload = {
        "model": model,
        "scaler": scaler,
        "feature_version": FEATURE_VERSION,
        "saved_at": datetime.utcnow().isoformat() + "Z",
    }
    if extra_meta:
        payload.update(extra_meta)
    joblib.dump(payload, p)
    return str(p)
