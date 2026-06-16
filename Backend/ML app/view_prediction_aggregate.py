# ML_Apps/views_prediction_aggregate.py
from typing import Dict, Any, List, Tuple
from bson import ObjectId
from collections import defaultdict

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .pymongo_client import get_db
from .prediction_state_pipeline import resolve_state_oid, build_state_dataframe
from .prediction_pipeline import run_prediction_pipeline, to_api_payload

# ---------- helpers (labels & aggregation) ----------

def _objid_str(oid: ObjectId) -> str:
    return str(oid) if isinstance(oid, ObjectId) else str(oid)

def _load_student_region_map(db, state_oid: ObjectId) -> Dict[str, Dict[str, str]]:
    """
    Returns { str(StudentID): { "District": str(oid) or "", "Taluka": str(oid) or "", "City": str(oid) or "" } }
    Only students in this state.
    """
    mapping: Dict[str, Dict[str, str]] = {}
    cur = db["students"].find(
        {"State": state_oid},  # students.State is ObjectId ref:contentReference[oaicite:3]{index=3}
        {"_id": 1, "District": 1, "Taluka": 1, "City": 1},
    )
    for s in cur:
        sid = _objid_str(s["_id"])
        mapping[sid] = {
            "District": _objid_str(s.get("District") or "") if s.get("District") else "",
            "Taluka": _objid_str(s.get("Taluka") or "") if s.get("Taluka") else "",
            "City": _objid_str(s.get("City") or "") if s.get("City") else "",
        }
    return mapping

def _fetch_labels(db, coll: str, ids: List[str]) -> Dict[str, str]:
    """
    Fetch human labels for region documents.
    We try common fields: 'district', 'taluka', 'city', 'name' (case preserved in your models).
    """
    obj_ids = [ObjectId(i) for i in ids if i]
    if not obj_ids:
        return {}
    label_map: Dict[str, str] = {}
    for doc in db[coll].find({"_id": {"$in": obj_ids}}):
        # try common fields
        label = (
            doc.get("district") or
            doc.get("taluka") or
            doc.get("city") or
            doc.get("name") or
            f"{coll[:-1].title()}-{str(doc['_id'])[-4:]}"
        )
        label_map[str(doc["_id"])] = str(label)
    return label_map

def _aggregate_by_key(
    results: List[Dict[str, Any]],
    student_region_map: Dict[str, Dict[str, str]],
    group_key: str,            # "District" | "Taluka" | "City"
    labels_map: Dict[str, str],
    with_top: bool = True,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """
    Group API student results by group_key using student_region_map.
    results[i] has: StudentID, StudentLabel, Risk_Score, Risk_Level, Dropout_Probability, ...
    """
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        sid = r.get("StudentID")
        rid = student_region_map.get(sid, {}).get(group_key) if sid else ""
        if rid:
            buckets[rid].append(r)

    summary = []
    for rid, rows in buckets.items():
        total = len(rows)
        high = sum(1 for x in rows if x.get("Risk_Level") == "High")
        med  = sum(1 for x in rows if x.get("Risk_Level") == "Medium")
        low  = sum(1 for x in rows if x.get("Risk_Level") == "Low")
        avg_score = float(sum(x.get("Risk_Score", 0.0) for x in rows) / max(total, 1))
        avg_prob  = float(sum(x.get("Dropout_Probability", 0.0) for x in rows) / max(total, 1))

        # Top N by probability, then score
        top_rows = sorted(
            rows,
            key=lambda z: (float(z.get("Dropout_Probability", 0.0)), float(z.get("Risk_Score", 0.0))),
            reverse=True,
        )[: top_n if with_top else 0]
        top_compact = [
            {
                "StudentLabel": t.get("StudentLabel"),
                "Risk_Level": t.get("Risk_Level"),
                "Risk_Score": float(t.get("Risk_Score", 0.0)),
                "Dropout_Probability": float(t.get("Dropout_Probability", 0.0)),
            } for t in top_rows
        ]

        summary.append({
            "group_id": rid,
            "group_label": labels_map.get(rid, f"{group_key}-{rid[-4:]}"),
            "counts": {"total": total, "high": high, "medium": med, "low": low},
            "avg_risk_score": round(avg_score, 2),
            "avg_dropout_probability": round(avg_prob, 4),
            "top_students": top_compact,
        })

    # deterministic order: high-risk first
    summary.sort(key=lambda s: (s["counts"]["high"], s["avg_dropout_probability"]), reverse=True)
    return summary


# ---------- views ----------
class PredictStateAggregateDistrictsView(APIView):
    """
    POST /ML_api/predict/state/aggregate/districts
    Body:
      {
        "state_name": "Rajasthan",
        "fees_months_denom": 12,       # optional
        "with_top_students": true,     # optional; default true
        "top_n": 5,                    # optional
        "force_retrain": false         # optional; use cached model if available
      }
    """
    def post(self, request):
        if "state_id" in request.data:
            return Response({"detail": "Do not send ObjectId. Use state_name."}, status=400)

        state_name = request.data.get("state_name")
        if not state_name:
            return Response({"detail": "state_name is required"}, status=400)

        with_top = bool(request.data.get("with_top_students", True))
        top_n = int(request.data.get("top_n", 5))
        force = bool(request.data.get("force_retrain", False))
        try:
            denom = int(request.data.get("fees_months_denom", 12))
        except Exception:
            denom = 12

        try:
            db = get_db()
            state_oid = resolve_state_oid(db, state_name=state_name)

            # 1) Build & predict once for the whole state
            df_state = build_state_dataframe(db, state_oid, fees_months_denom=denom)
            if df_state.empty:
                return Response({"groups": [], "total_students": 0}, status=200)

            scope_key = f"state:{str(state_oid)}"
            df_out = run_prediction_pipeline(df_state, model_scope=scope_key, force_retrain=force)
            payload = to_api_payload(df_out)  # {count, results}

            # 2) Attach region mapping & labels (District)
            student_region_map = _load_student_region_map(db, state_oid)
            distinct_ids = sorted({ student_region_map.get(r["StudentID"],{}).get("District","") for r in payload["results"] if student_region_map.get(r["StudentID"]) })
            labels = _fetch_labels(db, "districts", [x for x in distinct_ids if x])

            # 3) Aggregate
            groups = _aggregate_by_key(payload["results"], student_region_map, "District", labels, with_top, top_n)
            return Response({"state": state_name, "group_by": "district", "total_students": payload["count"], "groups": groups}, status=200)

        except Exception as e:
            return Response({"detail": f"Aggregation failed: {e}"}, status=500)


class PredictStateAggregateTalukasView(APIView):
    """
    POST /ML_api/predict/state/aggregate/talukas
    Body: same as /districts
    """
    def post(self, request):
        if "state_id" in request.data:
            return Response({"detail": "Do not send ObjectId. Use state_name."}, status=400)

        state_name = request.data.get("state_name")
        if not state_name:
            return Response({"detail": "state_name is required"}, status=400)

        with_top = bool(request.data.get("with_top_students", True))
        top_n = int(request.data.get("top_n", 5))
        force = bool(request.data.get("force_retrain", False))
        try:
            denom = int(request.data.get("fees_months_denom", 12))
        except Exception:
            denom = 12

        try:
            db = get_db()
            state_oid = resolve_state_oid(db, state_name=state_name)

            df_state = build_state_dataframe(db, state_oid, fees_months_denom=denom)
            if df_state.empty:
                return Response({"groups": [], "total_students": 0}, status=200)

            scope_key = f"state:{str(state_oid)}"
            df_out = run_prediction_pipeline(df_state, model_scope=scope_key, force_retrain=force)
            payload = to_api_payload(df_out)

            student_region_map = _load_student_region_map(db, state_oid)
            distinct_ids = sorted({ student_region_map.get(r["StudentID"],{}).get("Taluka","") for r in payload["results"] if student_region_map.get(r["StudentID"]) })
            labels = _fetch_labels(db, "talukas", [x for x in distinct_ids if x])
            groups = _aggregate_by_key(payload["results"], student_region_map, "Taluka", labels, with_top, top_n)
            return Response({"state": state_name, "group_by": "taluka", "total_students": payload["count"], "groups": groups}, status=200)

        except Exception as e:
            return Response({"detail": f"Aggregation failed: {e}"}, status=500)


class PredictStateAggregateCitiesView(APIView):
    """
    POST /ML_api/predict/state/aggregate/cities
    Body: same as /districts
    """
    def post(self, request):
        if "state_id" in request.data:
            return Response({"detail": "Do not send ObjectId. Use state_name."}, status=400)

        state_name = request.data.get("state_name")
        if not state_name:
            return Response({"detail": "state_name is required"}, status=400)

        with_top = bool(request.data.get("with_top_students", True))
        top_n = int(request.data.get("top_n", 5))
        force = bool(request.data.get("force_retrain", False))
        try:
            denom = int(request.data.get("fees_months_denom", 12))
        except Exception:
            denom = 12

        try:
            db = get_db()
            state_oid = resolve_state_oid(db, state_name=state_name)

            df_state = build_state_dataframe(db, state_oid, fees_months_denom=denom)
            if df_state.empty:
                return Response({"groups": [], "total_students": 0}, status=200)

            scope_key = f"state:{str(state_oid)}"
            df_out = run_prediction_pipeline(df_state, model_scope=scope_key, force_retrain=force)
            payload = to_api_payload(df_out)

            student_region_map = _load_student_region_map(db, state_oid)
            distinct_ids = sorted({ student_region_map.get(r["StudentID"],{}).get("City","") for r in payload["results"] if student_region_map.get(r["StudentID"]) })
            labels = _fetch_labels(db, "cities", [x for x in distinct_ids if x])
            groups = _aggregate_by_key(payload["results"], student_region_map, "City", labels, with_top, top_n)
            return Response({"state": state_name, "group_by": "city", "total_students": payload["count"], "groups": groups}, status=200)

        except Exception as e:
            return Response({"detail": f"Aggregation failed: {e}"}, status=500)
