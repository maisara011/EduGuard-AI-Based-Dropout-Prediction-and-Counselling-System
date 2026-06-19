import os
from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "eduguard-secret-key-2024")

DEMO_PASSWORD = "student123"

VALID_ROLES = {
    "administrator": {"label": "Administrator",      "icon": "shield"},
    "ai_dashboard":  {"label": "AI Dashboard",       "icon": "cpu"},
    "mentor":        {"label": "Mentor / Counselor", "icon": "users"},
}

# ── Train LogisticRegression model ──────────────────────────────
np.random.seed(42)
n = 500
attendance   = np.random.uniform(0, 100, n)
marks        = np.random.uniform(0, 100, n)
income       = np.random.randint(0, 3, n)
prev_dropout = np.random.randint(0, 2, n)

risk_score = (
    -0.04 * attendance
    - 0.03 * marks
    - 0.5  * income
    + 1.5  * prev_dropout
    + 4.0
    + np.random.normal(0, 0.5, n)
)
labels = (risk_score > 3.5).astype(int)

X_train = np.column_stack([attendance, marks, income, prev_dropout])
model = LogisticRegression(max_iter=1000)
model.fit(X_train, labels)


def predict_dropout(attendance_pct, marks_pct, family_income, prev_dropout_hist):
    income_lower  = family_income.lower()
    dropout_lower = prev_dropout_hist.lower()

    income_map  = {"low": 0, "medium": 1, "high": 2}
    dropout_map = {"yes": 1, "no": 0}
    income_val  = income_map.get(income_lower, 1)
    dropout_val = dropout_map.get(dropout_lower, 0)
    X           = np.array([[attendance_pct, marks_pct, income_val, dropout_val]])
    proba       = model.predict_proba(X)[0]
    ml_high_prob = float(proba[1])

    # ── Rule-based overrides (higher priority than ML model) ──────
    # Trigger High Risk if any critical threshold is breached:
    #   • Attendance < 40%
    #   • Marks < 35%
    #   • Family Income = Low AND Prior Dropout = Yes
    rule_triggered = (
        attendance_pct < 40
        or marks_pct < 35
        or (income_lower == "low" and dropout_lower == "yes")
    )

    if rule_triggered:
        # Clamp confidence floor to 82%; use model's value if it's already higher
        confidence = round(max(ml_high_prob, 0.82) * 100, 1)
        return "High Risk", confidence

    # ── Standard ML path ─────────────────────────────────────────
    prediction = model.predict(X)[0]
    confidence = round(float(max(proba)) * 100, 1)
    risk       = "High Risk" if prediction == 1 else "Low Risk"
    return risk, confidence


# ── Routes ──────────────────────────────────────────────────────

@app.route("/")
def root():
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        password = request.form.get("password", "")
        if password == DEMO_PASSWORD:
            session["authenticated"] = True
            return redirect(url_for("roles"))
        else:
            error = "Invalid password. Please try again."
    return render_template("login.html", error=error)


@app.route("/roles")
def roles():
    if not session.get("authenticated"):
        return redirect(url_for("login"))
    return render_template("roles.html")


@app.route("/dashboard/<role>")
def dashboard(role):
    if not session.get("authenticated"):
        return redirect(url_for("login"))
    if role not in VALID_ROLES:
        return redirect(url_for("roles"))
    session["role"] = role
    role_info = VALID_ROLES[role]
    return render_template("dashboard.html", role=role, role_label=role_info["label"])


@app.route("/predict", methods=["GET"])
def index():
    if not session.get("authenticated"):
        return redirect(url_for("login"))
    role = session.get("role", "ai_dashboard")
    return render_template(
        "index.html",
        role=role,
        role_label=VALID_ROLES.get(role, {}).get("label", ""),
    )


@app.route("/result", methods=["POST"])
def result():
    if not session.get("authenticated"):
        return redirect(url_for("login"))
    try:
        attendance   = float(request.form.get("attendance", 0))
        marks        = float(request.form.get("marks", 0))
        income       = request.form.get("income", "medium")
        prev_dropout = request.form.get("prev_dropout", "no")

        risk, confidence = predict_dropout(attendance, marks, income, prev_dropout)
        role = session.get("role", "ai_dashboard")

        return render_template(
            "result.html",
            risk=risk,
            confidence=confidence,
            attendance=attendance,
            marks=marks,
            income=income.capitalize(),
            prev_dropout=prev_dropout.capitalize(),
            role=role,
            role_label=VALID_ROLES.get(role, {}).get("label", ""),
        )
    except Exception:
        return redirect(url_for("index"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
