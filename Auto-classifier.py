import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from clover.over_sampling import ClusterOverSampler
from clover.distribution import DensityDistributor
from pycaret.classification import *
import shap
import io
import joblib
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px



# Load the PyCaret model
model = load_model('classifier-pipeline')


def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Layers'][0]
    return predictions


def run():
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app is created to classify a soft ground tunnel lithology')

# Create inputs for the different features of the data
    if add_selectbox == 'Online':
        pressure_gauge1 = st.number_input('Pressure gauge 1 (kPa)', min_value=0.0, max_value=210.27, value=64.95)
        pressure_gauge2 = st.number_input('Pressure gauge 2 (kPa)', min_value=0.0, max_value=259.0, value=60.85)
        pressure_gauge3 = st.number_input('Pressure gauge 3 (kPa)', min_value=0.0, max_value=175.36, value=58.52)
        pressure_gauge4 = st.number_input('Pressure gauge 4 (kPa)', min_value=0.0, max_value=426.93, value=67.23)
        digging_velocity_left = st.number_input('Digging velocity left (mm/min)', min_value=0.0, max_value=336.0, value=13.75)
        digging_velocity_right = st.number_input('Digging velocity right (mm/min)', min_value=0.0, max_value=239.0, value=13.27)
        shield_jack_stroke_left = st.number_input('Shield jack stroke left (mm)', min_value=0.0, max_value=3504.20, value=694.17)
        shield_jack_stroke_right = st.number_input('Shield jack stroke right (mm)', min_value=0.0, max_value=5946.13, value=696.96)
        propulsion_pressure = st.number_input('Propulsion pressure (MPa)', min_value=-81.87, max_value=31.55, value=15.60)
        total_thrust = st.number_input('Total thrust (kN)', min_value=-19870.27, max_value=7160.73, value=3424.15)
        cutter_torque = st.number_input('Cutter torque (kNm)', min_value=0.0, max_value=694.91, value=482.25)
        cutterhead_rotation_speed = st.number_input('Cutterhead rotation speed (rpm)', min_value=0.0, max_value=7.00, value=1.97)
        screw_pressure = st.number_input('Screw pressure (MPa)', min_value=-114.64, max_value=7.69, value=2.86)
        screw_rotation_speed = st.number_input('Screw rotation speed (rpm)', min_value=-1.22, max_value=78.25, value=0.90)
        gate_opening = st.number_input('Gate opening (%)', min_value=0.0, max_value=1.60, value=0.18)
        mud_injection_pressure = st.number_input('Mud injection pressure (MPa)', min_value=0.0, max_value=58.32, value=16.33)
        add_mud_flow = st.number_input('Add mud flow (L/min)', min_value=0.0, max_value=560.64, value=96.99)
        back_in_injection_rate = st.number_input('Back in injection rate (%)', min_value=0.0, max_value=100.0, value=0.19)

        output=""

        input_dict = {
        'pressure_gauge1': pressure_gauge1,
        'pressure_gauge2': pressure_gauge2,
        'pressure_gauge3': pressure_gauge3,
        'pressure_gauge4': pressure_gauge4,
        'digging_velocity_left': digging_velocity_left,
        'digging_velocity_right': digging_velocity_right,
        'shield_jack_stroke_left': shield_jack_stroke_left,
        'shield_jack_stroke_right': shield_jack_stroke_right,
        'propulsion_pressure': propulsion_pressure,
        'total_thrust': total_thrust,
        'cutter_torque': cutter_torque,
        'cutterhead_rotation_speed': cutterhead_rotation_speed,
        'screw_pressure': screw_pressure,
        'screw_rotation_speed': screw_rotation_speed,
        'gate_opening': gate_opening,
        'mud_injection_pressure': mud_injection_pressure,
        'add_mud_flow': add_mud_flow,
        'back_in_injection_rate': back_in_injection_rate}
        
        input_df = pd.DataFrame([input_dict])
        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = '$' + str(output)

        st.success('The output is {}'.format(output))

        st.subheader("Classification Report")
        plot_model(model, plot='class_report',  plot_kwargs={'title': 'LightGBM Classifier Classification Report'}, display_format="streamlit")

        st.subheader("Confusion Matrix")
        plot_model(model, plot='confusion_matrix', plot_kwargs={'title': 'LightGBM Classifier Confusion Matrix'}, display_format="streamlit")

        st.subheader("Feature Importance")
        interpret_model(model, display_format="streamlit")

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()

