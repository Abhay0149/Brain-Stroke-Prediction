from flask import Flask, render_template, request
import pandas as pd
import joblib
import plotly.express as px

app = Flask(__name__)

# Load model
stroke_model = joblib.load("model.joblib")

# Prediction function
def predict_input(single_input):
    input_df = pd.DataFrame([single_input])

    encoded_cols = stroke_model["encoded_cols"]
    numeric_cols = stroke_model["numeric_cols"]
    preprocessor = stroke_model["preprocessor"]

    # ⚠️ Safe transform
    input_df[encoded_cols] = preprocessor.transform(input_df)

    X = input_df[numeric_cols + encoded_cols]
    prediction = stroke_model["model"].predict(X)

    return prediction


# 🔥 GRAPH FUNCTION (2 graphs only)
def create_graphs():
    df = pd.read_csv("train.csv")

    # 1️⃣ Age vs Stroke
    fig1 = px.histogram(
        df,
        x="age",
        color="stroke",
        title="Age Distribution by Stroke"
    )
    fig1.update_layout(title_x=0.3)

    # 2️⃣ BMI vs Stroke
    fig2 = px.violin(
        df,
        y="bmi",
        x="stroke",
        box=True,
        title="BMI Distribution by Stroke"
    )
    fig2.update_layout(title_x=0.3)

    return (
        fig1.to_html(full_html=False),
        fig2.to_html(full_html=False)
    )


# 🔥 HOME PAGE
@app.route("/")
def home():
    return render_template("home.html")


# 🔥 PREDICTION PAGE
@app.route("/predict", methods=["GET", "POST"])
def predict():
    graph1, graph2 = create_graphs()

    if request.method == "POST":
        try:
            # Get form data safely
            gender = request.form.get("gender", "").lower()
            age = int(request.form.get("age", 0))
            hypertension = int(request.form.get("hypertension", 0))
            heart_disease = int(request.form.get("heart_disease", 0))
            ever_married = request.form.get("ever_married", "").lower()
            work_type = request.form.get("work_type", "")
            residence_type = request.form.get("residence_type", "")
            avg_glucose_level = float(request.form.get("avg_glucose_level", 0))
            bmi = float(request.form.get("bmi", 0))
            smoking_status = request.form.get("smoking_status", "").lower()

            # Mapping
            work_type_mapping = {
                "Government job": "Govt_job",
                "Children": "children",
                "Never Worked": "Never_worked",
                "Private": "Private",
                "Self-employed": "Self-employed"
            }

            single_input = {
                "gender": gender,
                "age": age,
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "ever_married": ever_married,
                "work_type": work_type_mapping.get(work_type, work_type),
                "Residence_type": residence_type,
                "avg_glucose_level": avg_glucose_level,
                "bmi": bmi,
                "smoking_status": smoking_status,
            }

            prediction = predict_input(single_input)[0]

            return render_template(
                "predict.html",
                prediction=prediction,
                form_data=request.form,
                graph1=graph1,
                graph2=graph2
            )

        except Exception as e:
            return f"Error: {e}"

    return render_template(
        "predict.html",
        prediction=None,
        form_data=None,
        graph1=graph1,
        graph2=graph2
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
