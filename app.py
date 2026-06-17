from flask import Flask, render_template, request
from sklearn.tree import DecisionTreeClassifier
import numpy as np

app = Flask(__name__)

# Sample data train panrom
X = [[85,90,1], [45,60,0], [30,50,0], [75,85,1], [50,65,1]]
y = [0, 1, 1, 0, 1] # 0=Low Risk, 1=High Risk
model = DecisionTreeClassifier().fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']
    marks = int(request.form['marks'])
    attendance = int(request.form['attendance'])
    fees = int(request.form['fees'])

    prediction = model.predict([[marks, attendance, fees]])[0]

    if prediction == 1:
        result = f"⚠️ HIGH RISK - {name}"
        suggestion = "Counseling, Extra class, Parent call recommended"
        color = "danger"
    else:
        result = f"✅ LOW RISK - {name}"
        suggestion = "Student is performing well"
        color = "success"

    return render_template('index.html', result=result, suggestion=suggestion, color=color)

if __name__ == '__main__':
    app.run(debug=True)
