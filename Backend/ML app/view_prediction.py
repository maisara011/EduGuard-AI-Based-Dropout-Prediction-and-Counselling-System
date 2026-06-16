from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from .summary import generate_gemini_insights 

from .pymongo_client import get_db
from .prediction_pipeline import (
    build_school_dataframe, run_prediction_pipeline, to_api_payload, resolve_school_oid
)

class PredictSchoolView(APIView):
    """
    POST /ML_api/predict/
    Body:
      { "school_public_id": <number> } OR { "school_name": "<string>" }
      + optional:
          { "fees_months_denom": 12,
            "with_gemini": true,              // default False
            "gemini_max_students": 10,        // optional
            "gemini_max_chars": 1800          // optional
           }
    NOTE: Raw ObjectIds from client are rejected by design.
    """

    def post(self, request):
        # Hard-rule: never accept ObjectId param from client
        if "school_id" in request.data:
            return Response(
                {"detail": "Do not send ObjectId. Use school_public_id (number) or school_name."},
                status=status.HTTP_400_BAD_REQUEST
            )

        school_public_id = request.data.get("school_public_id")
        school_name = request.data.get("school_name")
        if school_public_id is None and not school_name:
            return Response(
                {"detail": "Provide school_public_id (number) or school_name."},
                status=status.HTTP_400_BAD_REQUEST
            )
            
        with_gemini = bool(request.data.get("with_gemini", False))
        g_max_students = int(request.data.get("gemini_max_students", 10))
        g_max_chars = int(request.data.get("gemini_max_chars", 1800))

        try:
            denom = int(request.data.get("fees_months_denom", 12))
        except Exception:
            denom = 12

        try:
            db = get_db()
            school_oid = resolve_school_oid(db, school_public_id=school_public_id, school_name=school_name)
            df = build_school_dataframe(db, school_oid, fees_months_denom=denom)
            if df.empty:
                return Response({"count": 0, "results": []}, status=status.HTTP_200_OK)

            scope_key = f"school:{str(school_oid)}"
            force = bool(request.data.get("force_retrain", False))
            df_out = run_prediction_pipeline(df, model_scope=scope_key, force_retrain=force)            
            
            payload = to_api_payload(df_out)
            
            # Gemini summary
            if with_gemini:
                insights, g_status = generate_gemini_insights(
                    results=payload["results"],
                    school_label=school_name or (
                        f"SchoolID {school_public_id}" if school_public_id is not None else "this school"
                    ),
                    max_students=g_max_students,
                    max_chars=g_max_chars,
                )
                payload["gemini"] = {"status": g_status, "insights": insights or ""}

            return Response(payload, status=status.HTTP_200_OK)


        except Exception as e:
            return Response({"detail": f"Prediction failed: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
