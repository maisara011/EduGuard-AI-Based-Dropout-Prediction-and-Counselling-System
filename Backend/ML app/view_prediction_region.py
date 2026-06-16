# ML_Apps/views_prediction_region.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .pymongo_client import get_db
from .summary import generate_gemini_insights 

# Resolve + build for regions
from .prediction_region_pipeline import (
    resolve_district_oid,
    resolve_taluka_oid,
    resolve_city_oid,
    _build_region_dataframe,
)

# Core ML pipeline + payload utils (with caching)
from .prediction_pipeline import run_prediction_pipeline, to_api_payload


class PredictDistrictView(APIView):
    """
    POST /ML_api/predict/district/
    Body:
      {
        "district_name": "Jaipur",
        "fees_months_denom": 12,        # optional
        "with_gemini": true,            # optional
        "gemini_max_students": 10,      # optional
        "gemini_max_chars": 1800,       # optional
        "force_retrain": false          # optional
      }
    """

    def post(self, request):
        if "district_id" in request.data:
            return Response({"detail": "Do not send ObjectId. Use district_name."}, status=status.HTTP_400_BAD_REQUEST)

        district_name = request.data.get("district_name")
        if not district_name:
            return Response({"detail": "district_name is required"}, status=status.HTTP_400_BAD_REQUEST)

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
            district_oid = resolve_district_oid(db, district_name)
            df = _build_region_dataframe(db, {"District": district_oid}, fees_months_denom=denom)

            if df.empty:
                payload = {"count": 0, "results": []}
                if with_gemini:
                    payload["gemini"] = {"status": "ok", "insights": ""}
                return Response(payload, status=status.HTTP_200_OK)

            scope_key = f"district:{str(district_oid)}"
            df_out = run_prediction_pipeline(df, model_scope=scope_key, force_retrain=force)
            payload = to_api_payload(df_out)

            if with_gemini:
                insights, g_status = generate_gemini_insights(
                    results=payload["results"],
                    school_label=district_name,
                    max_students=g_max_students,
                    max_chars=g_max_chars,
                )
                payload["gemini"] = {"status": g_status, "insights": insights or ""}

            return Response(payload, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"detail": f"District prediction failed: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class PredictTalukaView(APIView):
    """
    POST /ML_api/predict/taluka/
    Body:
      {
        "taluka_name": "Bassi",
        "fees_months_denom": 12,        # optional
        "with_gemini": true,            # optional
        "gemini_max_students": 10,      # optional
        "gemini_max_chars": 1800,       # optional
        "force_retrain": false          # optional
      }
    """

    def post(self, request):
        if "taluka_id" in request.data:
            return Response({"detail": "Do not send ObjectId. Use taluka_name."}, status=status.HTTP_400_BAD_REQUEST)

        taluka_name = request.data.get("taluka_name")
        if not taluka_name:
            return Response({"detail": "taluka_name is required"}, status=status.HTTP_400_BAD_REQUEST)

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
            taluka_oid = resolve_taluka_oid(db, taluka_name)
            df = _build_region_dataframe(db, {"Taluka": taluka_oid}, fees_months_denom=denom)

            if df.empty:
                payload = {"count": 0, "results": []}
                if with_gemini:
                    payload["gemini"] = {"status": "ok", "insights": ""}
                return Response(payload, status=status.HTTP_200_OK)

            scope_key = f"taluka:{str(taluka_oid)}"
            df_out = run_prediction_pipeline(df, model_scope=scope_key, force_retrain=force)
            payload = to_api_payload(df_out)

            if with_gemini:
                insights, g_status = generate_gemini_insights(
                    results=payload["results"],
                    school_label=taluka_name,
                    max_students=g_max_students,
                    max_chars=g_max_chars,
                )
                payload["gemini"] = {"status": g_status, "insights": insights or ""}

            return Response(payload, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"detail": f"Taluka prediction failed: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class PredictCityView(APIView):
    """
    POST /ML_api/predict/city/
    Body:
      {
        "city_name": "Jaipur",
        "fees_months_denom": 12,        # optional
        "with_gemini": true,            # optional
        "gemini_max_students": 10,      # optional
        "gemini_max_chars": 1800,       # optional
        "force_retrain": false          # optional
      }
    """

    def post(self, request):
        if "city_id" in request.data:
            return Response({"detail": "Do not send ObjectId. Use city_name."}, status=status.HTTP_400_BAD_REQUEST)

        city_name = request.data.get("city_name")
        if not city_name:
            return Response({"detail": "city_name is required"}, status=status.HTTP_400_BAD_REQUEST)

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
            city_oid = resolve_city_oid(db, city_name)
            df = _build_region_dataframe(db, {"City": city_oid}, fees_months_denom=denom)

            if df.empty:
                payload = {"count": 0, "results": []}
                if with_gemini:
                    payload["gemini"] = {"status": "ok", "insights": ""}
                return Response(payload, status=status.HTTP_200_OK)

            scope_key = f"city:{str(city_oid)}"
            df_out = run_prediction_pipeline(df, model_scope=scope_key, force_retrain=force)
            payload = to_api_payload(df_out)

            if with_gemini:
                insights, g_status = generate_gemini_insights(
                    results=payload["results"],
                    school_label=city_name,
                    max_students=g_max_students,
                    max_chars=g_max_chars,
                )
                payload["gemini"] = {"status": g_status, "insights": insights or ""}

            return Response(payload, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"detail": f"City prediction failed: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
