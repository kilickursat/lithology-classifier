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


##########################################################
# Data loading options
data_load_option = st.radio("Data Load Option", ("Online", "Batch"))

if data_load_option == "Online":
  # Online data loading
  online_data = st.text_area("Enter data in CSV format")
  if st.button("Load Data"):
    try:
      df = pd.read_csv(io.StringIO(online_data))
    except Exception as e:
      st.write(f"Error loading data: {e}")
      df = None

else:
  # Batch data loading
  uploaded_file = st.file_uploader("classification_model.xlsx", type="xlsx")
  if uploaded_file is not None:
    try:
      df = pd.read_excel(uploaded_file)
    except Exception as e:
      st.write(f"Error loading data: {e}")
      df = None

if df is not None:
  # Do something with the data
  pass
#########################################################
"""
# Data loading options
data_load_option = st.radio("Data Load Option", ("Online", "Batch"))

if data_load_option == "Online":
    # Online data loading
    online_data = st.text_area("Enter data in CSV format")
    if st.button("Load Data"):
        df = pd.read_csv(io.StringIO(online_data))

else:
    # Batch data loading
    uploaded_file = st.file_uploader("classification_model.xlsx", type="xlsx")
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)

# Define a function to make predictions
def predict(input_data):
  try:
    predictions = predict_model(estimator=model, data=input_data)
    return predictions['Label'][0]
  except Exception as e:
    raise st.ScriptRunner.StopExecution(e)
"""
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
        
        # Create inputs for the different features of the data
        output_lithology = st.selectbox('Output lithology', ['VSC', 'VG', 'VGS'])
        
        # Display a loading indicator when the user clicks the "Predict" button
        with st.spinner('Predicting lithology class...'):
          if st.button('Predict'):
            # Make a prediction
            prediction = predict({'output_lithology': output_lithology})
        
            # Display the prediction to the user
            st.write('Predicted lithology class:', prediction)

# Calculate and display the confusion matrix
st.write("Classification report")
plot_model(model,plot='class_report',  plot_kwargs = {'title' : 'LightGBM Classifier Classification Report'},display_format="streamlit")
st.write("Confusion matrix")
plot_model(model,plot='confusion_matrix',  plot_kwargs = {'title' : 'LightGBM Classifier Confusion Matrix'},display_format="streamlit")
st.write("Feature Importance:")
interpret_model(model,display_format="streamlit")


