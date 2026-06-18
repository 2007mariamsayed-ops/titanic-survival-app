
import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('titanic_model.pkl')

st.title("🚢 Titanic Survival Predictor")

st.write("Enter passenger details:")

# -------------------
# Inputs
# -------------------

age = st.number_input("Age", 0, 100, 25)

sex = st.selectbox("Sex", ["male", "female"])
sex = 1 if sex == "male" else 0

pclass = st.selectbox("Pclass", [1, 2, 3])

sibsp = st.number_input("Siblings/Spouses aboard", 0, 10, 0)

parch = st.number_input("Parents/Children aboard", 0, 10, 0)

fare = st.number_input("Fare", 0.0, 500.0, 50.0)

embarked = st.selectbox("Embarked", ["S", "C", "Q"])
if embarked == "S":
    embarked = 0
elif embarked == "C":
    embarked = 1
else:
    embarked = 2

# -------------------
# Prediction
# -------------------

if st.button("Predict"):

    input_data = np.array([[age, sex, pclass, sibsp, parch, fare, embarked]])

    proba = model.predict_proba(input_data)[0]

    not_survived = proba[0]
    survived = proba[1]

    st.subheader("📊 Result")

    st.write(f"❌ Not Survived: {not_survived:.2f}")
    st.write(f"🎉 Survived: {survived:.2f}")

    if survived > 0.5:
        st.success("High chance of survival 🎉")
    else:
        st.error("Low chance of survival 💀")

    st.progress(float(survived))
