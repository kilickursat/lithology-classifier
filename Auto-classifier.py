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

setup()

# Load the PyCaret model
model = load_model('classifier-pipeline')


def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Layers'][0]
    return predictions


# Create a Streamlit user interface
st.title('Tunnel Lithology Identification Classifier')
def run():
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app is created to classify a soft ground tunnel lithology')

# Create inputs for the different features of the data
def user_input_features():
    if add_selectbox == 'Batch':
        pressure_gauge1 = st.number_input('Pressure gauge 1 (kPa)', min_value=float(df['pressure_gauge1'].min()), value=0)
        pressure_gauge2 =st.number_input('Pressure gauge 2 (kPa)', min_value=float(df['pressure_gauge2'].min()), value=0)
        pressure_gauge3 = st.number_input('Pressure gauge 3 (kPa)', min_value=float(df['pressure_gauge3'].min()), value=0)
        pressure_gauge4 =st.number_input('Pressure gauge 4 (kPa)', min_value=float(df['pressure_gauge4'].min()), value=0)
        digging_velocity_left = st.number_input('Digging velocity left (mm/min)', min_value=float(df['digging_velocity_left'].min()), value=0)
        digging_velocity_right = st.number_input('Digging velocity right (mm/min)', min_value=float(df['digging_velocity_right'].min()), value=0)
        shield_jack_stroke_left = st.number_input('Shield jack stroke left (mm)', min_value=float(df['shield_jack_stroke_left'].min()), value=0)
        shield_jack_stroke_right = st.number_input('Shield jack stroke right (mm)', min_value=float(df['shield_jack_stroke_right'].min()), value=0)
        propulsion_pressure = st.number_input('Propulsion pressure (MPa)', min_value=float(df['propulsion_pressure'].min()), value=0)
        total_thrust = st.number_input('Total thrust (kN)', min_value=float(df['total_thrust'].min()), value=0)
        cutter_torque = st.number_input('Cutter torque (kNm)', min_value=float(df['cutter_torque'].min()), value=0)
        cutterhead_rotation_speed = st.number_input('Cutterhead rotation speed (rpm)', min_value=float(df['cutterhead_rotation_speed'].min()), value=0)
        screw_pressure = st.number_input('Screw pressure (MPa)', min_value=float(df['screw_pressure'].min()), value=0)
        screw_rotation_speed = st.number_input('Screw rotation speed (rpm)', min_value=float(df['screw_rotation_speed'].min()), value=0)
        gate_opening =st.number_input('Gate opening (%)', min_value=float(df['gate_opening'].min()), max_value=100, value=0)
        mud_injection_pressure = st.number_input('Mud injection pressure (MPa)', min_value=float(df['mud_injection_pressure'].min()), value=0)
        add_mud_flow = st.number_input('Add mud flow (L/min)', min_value=float(df['add_mud_flow'].min()), value=0)
        back_in_injection_rate = st.number_input('Back in injection rate (%)', min_value=float(df['back_in_injection_rate'].min()), max_value=100, value=0)
        """
        tunnel_depth = st.number_input('Tunnel depth (m)', 0.0, 100.0, 50.0)
        tunnel_diameter = st.slider('Tunnel diameter (m)', 0.0, 100.0, 50.0)
        rock_type = st.selectbox('Rock type', ['sandstone', 'limestone', 'shale'])
        """
        
        output_dict = {'VCS': 0, 'VG': 1,'VSG': 2}
        output_df = pd.DataFrame([output_dict])


        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = '$' + str(output)

        st.success('The output is {}'.format(output))

        # Calculate and display the confusion matrix
        st.subheader("Classification report")
        plot_model(model,plot='class_report',  plot_kwargs = {'title' : 'LightGBM Classifier Classification Report'},display_format="streamlit")
        st.subheader("Confusion matrix")
        plot_model(model,plot='confusion_matrix',  plot_kwargs = {'title' : 'LightGBM Classifier Confusion Matrix'},display_format="streamlit")
        st.subheader("Feature Importance:")
        interpret_model(model,display_format="streamlit")



    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv","xlsx"])

        if file_upload is not None:
            if file_upload.type == 'application/vnd.ms-excel':  # Check if the uploaded file is in Excel format
                data = pd.read_excel(file_upload)
            else:
                data = pd.read_csv(file_upload)
            
            data = data.dropna()
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)
          

if __name__ == '__main__':
    run()


