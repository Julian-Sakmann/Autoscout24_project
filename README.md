# Autoscout 24 Project

![Python](https://img.shields.io/badge/Python-3.11.5-blue)
![Framework](https://img.shields.io/badge/Streamlit-1.27-yellow)

This is a demo project where I prepare, analyse and visualise car sales data from Autoscout24.de.
Moreover, I have employed Machine-Learning models to predict car prices based on the car-related features.
An interactive streamlit app rounds of the project by presenting the results and allows the user to make some car price predictions based on the models I have estimated.

The project is structured in three parts:

1) Data Exploration and Data Preperation.
2) ML Model Estimation
3) Streamlit App

## Requirements

In order to run the Streamlit App you will need:
- An IDE with Python (Version 3.11 or higher)
- Jupyter

## How to run the App

### Clone the Autoscout24_project repository on your computer:

```bash
git clone https://github.com/Julian-Sakmann/Autoscout24_project.git
```
### Open and run the segments in autoscout24_data_exploration.ipynb
This will create the plots and perform the data cleaning/preperation steps.
The cleaned data will then be saved to the project folder for later use.

### Open and run the autoscout24_model_estimation.py script
This will estimate the ML Models and create an joblib-dump file "trained_models.joblib".
The dump-file contains the estimated ML-models which will then be imported and deployed in the streamlit-app

### Open your command prompt and change the directory to the project folder on your PC (example):
```
cd "C:\Users\Anwender\Desktop\Autoscout24_project"
```

### Run the App (it will open in your default browser):
```
streamlit run autoscout24_streamlit_app.py
```
























