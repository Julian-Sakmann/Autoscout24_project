## Autoscout24 Interactive Streamlit App

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import streamlit as st


###  to start the streamlit app:
    
#    open:  Anaconda Powershell Prompt
#    type:  cd "C:\Users\Anwender\Desktop\Autoscout24_project"   # change the path to the cloned GitHub repository on your PC
#    type:  streamlit run autoscout24_streamlit.py


# Preliminary settings
# Set the path to the working directory. 
path = r"C:\Users\Anwender\Desktop\Autoscout24_project"
image_width = 600 
st.set_page_config(layout="wide")

# Load data
os.chdir(path)

df_cleaned  = pd.read_csv('cleaned_data.csv')
df_original = pd.read_csv('autoscout24.csv')


## Streamlit App
# Custom Markdown styling for the title
st.write("# <span style='font-size:60px; color: white; text-decoration:underline;'>Project: autoscout24</span>", unsafe_allow_html=True)
st.write("<span style='font-size:30px; color: white;'>In this Project we are going to analyze car-sales data from autoscout24. "
         "<br>We are also going to make predictions on car-prices using Machine Learning techniques!</span>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Navigation")

# Sidebar navigation
selected_page = st.sidebar.radio("Navigate to:", [
    "Number of Cars Sold",
    "Sales",
    "Brands (Quantity)", 
    "Brands (Prices)",
    "Features and Correlations",
    "Luxury Cars",
    "Machine Learning",
    "Car Price Prediction"
    ])












