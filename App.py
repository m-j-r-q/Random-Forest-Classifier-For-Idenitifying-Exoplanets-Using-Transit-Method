import streamlit as st
import joblib
import numpy as np
import pandas as pd

RFC = joblib.load("RandomForestClassifier.pkl")
Features = ["koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec", "koi_tce_plnt_num", "koi_fwm_stat_sig", "koi_dikco_msky", "koi_count", "koi_time0bk", "koi_model_snr"]
Encode = {"CANDIDATE":1, "FALSE POSITIVE":0}

st.title("Exoplanet Identification Using The Transit Method:")
st.write("The following model is a Random Forest Classifier, trained on the Cummulative Kepler Objects of Interest (KOI) dataset provided by the NASA Exoplanet Science Institute. The data set consists of over 9500 Treshold Crossing Events (TCEs) which have been classified as Potential Exoplanets or False Positives. You can add your own data in the input fields below to classify the TCE.")

st.divider()
st.subheader("Model Performance")
results_df = pd.DataFrame({
    "Metric": ["Accuracy", "AUC Score", "Precision"],
    "Value": [f"{0.993}", f"{0.998}", f"{0.99}"]
})
st.table(results_df) 
st.divider()

st.subheader("ROC Curve - Random Forest Classifier")
st.image("ROC_Curve.png", use_container_width=True)
st.divider()

st.subheader("Enter Variable Values:")
st.write("Enter the values for the following variables to compute the prediction. Please recheck that the values were entered correctly after you press enter:")
st.markdown("The explanation for each variable is given [here](https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html)")

inputs = []

for feature in Features:
    value = st.number_input(f"{feature}:", value=1.0, step=0.0001, format="%.4f")
    inputs.append(value)

if st.button("Predict"):
    input_data = np.array([inputs])
    prediction = RFC.predict(input_data)[0]
    probability = round(RFC.predict_proba(input_data)[0][1], 4)

    if prediction == 1:
        st.write("This Treshold Crossing Event (TCE) is a Potential Candidate.")
        st.write(f"The probability of the TCE being a Candidate is: {probability} ")
    elif prediction == 0:
        st.write("This Treshold Crossing Event (TCE) is a False Positive:")
        st.write(f"The probability of the TCE being a Candidate is: {probability} ")

    
