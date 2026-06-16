# ML_Apps/views_prediction_state.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .pymongo_client import get_db
from .summary import generate_gemini_insights

# Build + resolve for STATE
from .prediction_state_pipeline import resolve_state_oid, build_state_dataframe
# Core ML pipeline + payload utils (with caching)
from .prediction_pipeline import run_prediction_pipeline, to_api_payload


class PredictStateView(APIView):
    """
    POST /ML_api/predict/state/
    Body:
      {
        "state_name": "Rajasthan",
        "fees_months_denom": 12,        # optional, default 12
        "with_gemini": true,            # optional, default false
        "gemini_max_students": 10,      # optional
        "gemini_max_chars": 1800,       # optional
        "force_retrain": false          # optional, default false (use cached .pkl if present)
      }
    Rule: Never accept ObjectIds from client.
    """

    def post(self, request):
        # Never accept raw ObjectId
        if "state_id" in request.data:
            return Response(
                {"detail": "Do not send ObjectId. Use state_name (string)."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        state_name = request.data.get("state_name")
        if not state_name:
            return Response({"detail": "state_name is required."}, status=status.HTTP_400_BAD_REQUEST)

        with_gemini = bool(request.data.get("with_gemini", False))
        g_max_students = int(request.data.get("gemini_max_students", 10))
        g_max_chars = int(request.data.get("gemini_max_chars", 1800))
        force = bool(request.data.get("force_retrain", False))

        try:
            denom = int(request.data.get("fees_months_denom", 12))
        except Exception:
            denom = 12

        try:
            db = get_db()
            state_oid = resolve_state_oid(db, state_name=state_name)

            # Build DF for the entire STATE
            df = build_state_dataframe(db, state_oid, fees_months_denom=denom)
            if df.empty:
                payload = {"count": 0, "results": []}
                if with_gemini:
                    payload["gemini"] = {"status": "ok", "insights": ""}
                return Response(payload, status=status.HTTP_200_OK)

            # Train/Load cached model and predict
            scope_key = f"state:{str(state_oid)}"
            df_out = run_prediction_pipeline(df, model_scope=scope_key, force_retrain=force)
            payload = to_api_payload(df_out)

            # Optional Gemini summary
            if with_gemini:
                insights, g_status = generate_gemini_insights(
                    results=payload["results"],
                    school_label=state_name,
                    max_students=g_max_students,
                    max_chars=g_max_chars,
                )
                payload["gemini"] = {"status": g_status, "insights": insights or ""}

            return Response(payload, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"detail": f"State prediction failed: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
