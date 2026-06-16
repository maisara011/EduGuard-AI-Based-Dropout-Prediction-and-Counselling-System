# ML_Apps/prediction_state_pipeline.py
from typing import Dict, Any, List
from bson import ObjectId
from collections import defaultdict
import re
import numpy as np
import pandas as pd

# Reuse helpers & ML from your working school pipeline
from .prediction_pipeline import (
    _avg_test_score_from_marks,
    run_prediction_pipeline,
    to_api_payload,
)

# ---------- Resolve state by safe input (name only) ----------
def resolve_state_oid(db, *, state_name: str) -> ObjectId:
    """
    Resolve the state's Mongo _id using ONLY a human-friendly name (case-insensitive).
    Never accept ObjectIds from client.
    'states' collection stores: { name: <str> }  (per your schema)
    """
    if not state_name or not isinstance(state_name, str):
        raise ValueError("Provide state_name (string).")
    q = {"name": {"$regex": f"^{re.escape(state_name.strip())}$", "$options": "i"}}
    doc = db["states"].find_one(q, {"_id": 1})
    if not doc:
        raise ValueError("State not found for the provided name.")
    return doc["_id"]

# ---------- Build dataframe for ALL students in a state ----------
def build_state_dataframe(db, state_oid: ObjectId, fees_months_denom: int = 12) -> pd.DataFrame:
    """
    Assemble the minimal features for the notebook across a STATE:
      StudentID | StudentLabel | Attendance_Rate | Test_Score | Fees | Reason

    Collections / fields used:
      - students: AttendancePercentage, Reasons, RollNumber, Name, State (ObjectId)  :contentReference[oaicite:3]{index=3}
      - marks   : Students[].Student1, Students[].marks                              
      - fees    : Students[].student_id, Students[].No_unpaid_Month                  
    """
    # 1) Pull students for this state
    cur = db["students"].find(
        {"State": state_oid},  # students.State is ObjectId ref to states  :contentReference[oaicite:6]{index=6}
        {
            "_id": 1,
            "AttendancePercentage": 1,
            "Reasons": 1,
            "RollNumber": 1,
            "Name": 1,
        },
    )
    students = list(cur)
    if not students:
        return pd.DataFrame(columns=["StudentID", "StudentLabel", "Attendance_Rate", "Test_Score", "Fees", "Reason"])

    # Build presentable label (no ObjectIds exposed)
    def _mask_name(name: str) -> str:
        if not isinstance(name, str) or not name.strip():
            return ""
        parts = name.strip().split()
        if len(parts) == 1:
            word = parts[0]
            return word[:1].upper() + word[1:].lower()
        first, last = parts[0], parts[-1]
        return f"{first[:1].upper()}{first[1:].lower()} {last[:1].upper()}."

    student_ids = [s["_id"] for s in students]
    base: Dict[ObjectId, Dict[str, Any]] = {}
    for s in students:
        sid_obj = s["_id"]
        sid_str = str(sid_obj)
        roll = (s.get("RollNumber") or "").strip()
        name = (s.get("Name") or "").strip()
        if roll:
            label = str(roll)
        elif name:
            label = _mask_name(name)
        else:
            label = f"Student-{sid_str[-4:]}"

        base[sid_obj] = {
            "StudentID": sid_str,                 # keep internal id in API (remove if you want)
            "StudentLabel": label,                # safe label for display/Gemini
            "Attendance_Rate": float(s.get("AttendancePercentage", 0.0) or 0.0),
            "Reason": s.get("Reasons") or "",
        }

    # 2) Marks across the state: query all docs that include any of these students
    #    (faster than looping schools; matches nested Students[].Student1)  
    st_set = set(student_ids)
    marks_map: Dict[ObjectId, List[float]] = defaultdict(list)
    for doc in db["marks"].find({"Students.Student1": {"$in": student_ids}}, {"Students": 1}):
        for row in doc.get("Students", []):
            st_id = row.get("Student1")
            if st_id in st_set:
                m = row.get("marks") or {}
                avg = _avg_test_score_from_marks(m)
                if np.isfinite(avg):
                    marks_map[st_id].append(avg)

    for oid in base.keys():
        base[oid]["Test_Score"] = float(np.mean(marks_map[oid])) if marks_map.get(oid) else 0.0

    # 3) Fees across the state: match any fee doc that contains these students  
    fees_by_student: Dict[ObjectId, float] = {}
    for fdoc in db["fees"].find({"Students.student_id": {"$in": student_ids}}, {"Students": 1}):
        if not isinstance(fdoc.get("Students"), list):
            continue
        for s in fdoc["Students"]:
            st = s.get("student_id")
            if st not in st_set:
                continue
            months = s.get("No_unpaid_Month", 0)
            try:
                frac = float(months) / float(fees_months_denom or 12)
            except Exception:
                frac = 0.0
            fees_by_student[st] = max(fees_by_student.get(st, 0.0), max(0.0, min(1.0, frac)))

    for oid in base.keys():
        base[oid]["Fees"] = float(fees_by_student.get(oid, 0.0))

    # 4) Clean DF
    df = pd.DataFrame(base.values())
    df["Test_Score"] = df["Test_Score"].fillna(0.0)
    df["Attendance_Rate"] = df["Attendance_Rate"].fillna(0.0)
    df["Fees"] = df["Fees"].fillna(0.0)
    df["Reason"] = df["Reason"].fillna("")
    return df

# ---------- End-to-end for state ----------
def predict_for_state(db, state_oid: ObjectId, fees_months_denom: int = 12) -> Dict[str, Any]:
    df = build_state_dataframe(db, state_oid, fees_months_denom=fees_months_denom)
    if df.empty:
        return {"count": 0, "results": []}
    df_out = run_prediction_pipeline(df)
    return to_api_payload(df_out)
