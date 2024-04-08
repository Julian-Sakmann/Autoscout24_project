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
from joblib import load


###  to start the streamlit app:
    
#    open:  Anaconda Powershell Prompt
#    type:  cd "C:\Users\Anwender\Desktop\Autoscout24_project"   # change the path to the cloned GitHub repository on your PC
#    type:  streamlit run autoscout24_streamlit_app.py


####################################


# Preliminary settings
# Set the path to the working directory. 
path = r"C:\Users\Anwender\Desktop\Autoscout24_project"
os.chdir(path)
image_width = 600 
st.set_page_config(layout="wide")

# Load data
df  = pd.read_csv('cleaned_data.csv', index_col=False)
df = df.drop('Unnamed: 0', axis=1)
df_original = pd.read_csv('autoscout24.csv')

# Load the trained models
trained_models = load('trained_models.joblib')

# Extract the trained models from the loaded dictionary
lr = trained_models['lr_model']
dt_reg = trained_models['dt_reg_model']
rf_reg = trained_models['rf_reg_model']

# Load the data
X = df
X = X.drop(columns = ['marke', 'model', 'fuel', 'gear', 'offerType', 'year'])

# Cleaned DataFrame without Dummy Variables
df_no_dummy = df[['mileage', 'marke', 'model', 'fuel', 'gear', 'offerType', 'price', 'hp', 'year']]

# Extract the target 'price'
y = X.pop('price')

# Extract feature names for later use
feature_names = X.columns.tolist()


####################################

# Use the imported ML-Models for model assessment

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

# Linear Regression
y_predict_lr = lr.predict(X_test)

# Model evaluation: Linear Regression
r2_score_lr = metrics.r2_score(y_test, y_predict_lr)
mse_lr      = metrics.mean_squared_error(y_test, y_predict_lr)
mae_lr      = metrics.mean_absolute_error(y_test, y_predict_lr)

# Decision Tree Regression
y_predict_dt = dt_reg.predict(X_test)

# Model evaluation: Decision Tree Regression
r2_score_dt = metrics.r2_score(y_test, y_predict_dt)
mse_dt      = metrics.mean_squared_error(y_test, y_predict_dt)
mae_dt      = metrics.mean_absolute_error(y_test, y_predict_dt)

# Random Forest Regression
y_predict_rfreg = rf_reg.predict(X_test)

# Model evaluation: Random Forest Regression
r2_score_rfreg = metrics.r2_score(y_test, y_predict_rfreg)
mse_rfreg = metrics.mean_squared_error(y_test, y_predict_rfreg)
mae_rfreg = metrics.mean_absolute_error(y_test, y_predict_rfreg)


####################################


## Streamlit App


# Sidebar navigation
selected_page = st.sidebar.radio("Navigate to:", [
    "Intro Page",
    "Number of Cars Sold",
    "Sales",
    "Brands (Quantity)", 
    "Brands (Prices)",
    "Features and Correlations",
    "Machine Learning",
    "Car Price Prediction"
    ])



#Content Sections
if selected_page == "Intro Page":
    
    
    col1, col2 = st.columns([1, 1])  

    
    with col1:
        st.write("# <span style='font-size:60px; color: white; text-decoration:underline;'>Project: autoscout24</span>", unsafe_allow_html=True)
        st.write("<span style='font-size:30px; color: white;'>In this Project we are going to analyze car-sales data from autoscout24. "
                 "<br>We are also going to make predictions on car-prices using Machine Learning techniques!</span>", unsafe_allow_html=True)
    
    with col2:
        st.title("")
        st.image("Autoscout24_Logo.png", width=image_width)
        
elif selected_page == "Number of Cars Sold":
    st.title("Number of Cars Sold")
    st.header("How many cars have been sold in total? How many have been sold over the time?")

    # Use columns to display image, number, and text
    col1, col2 = st.columns([1, 1])  

    # Insert picture
    with col1:
        st.header("Quantity Of Cars Sold Per Year")
        st.image("number_of_sales_over_the_years.png", width=image_width)

    # Insert number and text to the right
    with col2:
        total_cars_sold = len(df)  
        
        st.markdown(f"<h2 style='color: white;'>Total Cars Sold: {total_cars_sold}</h2>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 16px;'>The quantity of cars sold is pretty constant over the years (between 4100-4300 cars are sold per year). However, we can see a relatively strong decline in sales from 2020 to 2021 due to the Coronavirus pandemic, which should be monitored closely in the near future.</p>", unsafe_allow_html=True)
          


elif selected_page == "Sales":  
    st.title("Sales")     
    
    col1, col2 = st.columns([1,1])
    
    with col1:
        st.header("Revenue from Car Sales per Year")
        st.image("total_sales_over_the_years.png", width=image_width)
        
    with col2:
        total_sales = y.sum()
        average_price = y.mean()
        st.title("")
        st.markdown(f"<h2 style='color: white;'>Total Sales: {round(total_sales / 1000,2)} mil.€</h2>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='color: white;'>Avg. Price per Car: {round(average_price * 1000,2)}€</h2>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 16px;'>Comparing the total sales over the years we can identify a steady increase in total revenue from year to year. The strongest increase happened between 2016 and 2020, where the total sales per year have more than doubled.</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 16px;'>Moreover, we can make out a small decline in total sales per year, matching the decline in the quantity of cars sold.</p>", unsafe_allow_html=True)
        
elif selected_page == "Brands (Quantity)":
    st.title("Brands (Quantity)")
    st.header("Which brands are in the data?")
    
    brands_df = pd.read_csv('brands_df.csv')
    brands_df.rename(columns={'Count': 'Cars Sold'}, inplace=True)
    #st.dataframe(brands_df, width=300)
    
    col1, col2 = st.columns([0.8, 1])
    
    
    with col1:
        st.dataframe(brands_df, height=300, width=300)

        
    with col2:
        st.markdown("<p style='font-size: 16px;'>Here we can get an overview on what brands are represented in our data. In this section we are taking a look at the number of cars that have been sold.</p>", unsafe_allow_html=True) 
        st.markdown("<p style='font-size: 16px;'>As expected we can identify the big car brands like Volkswagen, Opel, Ford, Skoda, Renault, Audi, BMW and Mercedes among the top selling brands in terms of number of cars sold in the plot on the left.</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 16px;'>Additionally I have also plotted the least selling brands within our data on the right side plot.</p>", unsafe_allow_html=True)
        
    col1, col2 = st.columns([0.8, 1])

    with col1:
        st.header("Top Selling Brands")
        st.image("top_selling_brands_plot.png", width=image_width)
        
    with col2:
        st.header("Least Selling Brands")
        st.image("least_selling_brands_plot.png", width=image_width)




elif selected_page == "Brands (Prices)":
    st.title("Brands (Prices)")
    st.header("Which brands has the highest average price?")
    
    
    col1, col2 = st.columns([0.8, 1])
    
    with col1:
        st.image("top_20_brands_avg_price.png", width=image_width)
        
    with col2:
        st.markdown("<p style='font-size: 16px;'>In this section we are looking at the average price for a car for a given brand</p>", unsafe_allow_html=True) 
        st.markdown("<p style='font-size: 16px;'>Looking at the plot to the left we can identify the more luxurious car brands such as Maybach, Ferrari and Lamborghini.</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 16px;'>It should be noted that the car prices heavily depend on weather or not a car has been used/owned before. This has not been taken into account while creating this graph. Given that the number of cars sold from the high-end brands is fairly small it is possible that the average price for a given brand is biased. For a more accurate depiction of the average price for a car from a given brand we could further disaggregate the cars within each brand by offertype (i.e. new car, used car, company owned car). </p>", unsafe_allow_html=True)



    # Get user input for the brand
    st.title("")
    st.subheader("Detailed Information on Prices for a given Brand")
    brands_list = df['marke'].unique()  # Get unique brands from the DataFrame
    user_brand = st.selectbox("Select a brand:", brands_list, index=brands_list.tolist().index("Porsche"))

    
    def get_brand_statistics(dataframe, user_brand):
        # Filter the DataFrame for the specified brand
        brand_data = dataframe[dataframe['marke'] == user_brand]
    
        # Check if the brand exists in the DataFrame
        if not brand_data.empty:
            # Calculate statistics for the specified brand
            average_price = brand_data['price'].mean()
            median_price = brand_data['price'].median()
            min_price = brand_data['price'].min()
            max_price = brand_data['price'].max()
            lower_percentile_25 = np.percentile(brand_data['price'], 25)
            upper_percentile_75 = np.percentile(brand_data['price'], 75)
            cars_sold = len(brand_data)
    
            # Output the results in the app
            st.markdown(f"**Price Statistics for {user_brand}:**", unsafe_allow_html=True)
            st.write(f"**Number of Cars Sold:** {cars_sold}")
            st.write(f"**Average Price:**      {average_price:.2f} Tsd.€")
            st.write(f"**Median Price:**       {median_price:.2f} Tsd.€")
            st.write(f"**Price Range:**        {min_price:.2f} - {max_price:.2f} Tsd.€")
            st.write(f"**25% Percentile:**     {lower_percentile_25:.2f} Tsd.€")
            st.write(f"**75% Percentile:**     {upper_percentile_75:.2f} Tsd.€")
        else:
            st.markdown(f"\n**The Brand '{user_brand}' is not found in the dataset.**")
    
    # Button to trigger the analysis
    if st.button("Get Brand Statistics"):
        # Call the function with user input
        get_brand_statistics(df_original, user_brand)


elif selected_page == "Features and Correlations":

    st.title("Features")
    st.dataframe(df_no_dummy)
          
    st.title("Correlations")
    st.header("How does the correlation between the (numeric) features look like?")
    
    col1, col2 = st.columns([0.8, 1])
    with col1: 
        st.image("correlations.png", width=image_width)
    
    with col2:
        st.write("We observe a robust correlation between horsepower (hp) and prices with a correlation coefficient of ρ=0.75, as expected. Additionally, a negative correlation between the mileage a car has accumulated and its price (ρ=-0.3) can be identified. This suggests that the price of a car tends to depreciate as its mileage increases, reflecting the impact of car usage on its value. Lastly, there is no statistically significant correlation between horsepower and mileage, indicating that these two factors do not have a strong linear relationship.")
    
    st.title("")
    st.header("Scatterplots")
    st.image("scatterplots.png", width=image_width)


elif selected_page == "Machine Learning":
    
    st.title("Machine Learning")
    st.write("In this section I am going to apply machine learning techniques in order to predict car prices.")
    st.write("In preperation of this task all of the ctegorical features have been converted into binary-variables using one-hot-encoding.")
    st.write("")
    st.subheader("Data after encoding:")
    X_subset = X.head(100)
    st.dataframe(X_subset)
    st.write("")
    
    st.subheader("Preperation")
    st.write("Split the data in training and test sets")
    st.code("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)", language='python')
    

    
    st.subheader("Linear Regression")
    st.write("Fitting the Model")
    st.code("""
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            y_predict = lr.predict(X_test)""", language='python')
            
    
    st.subheader("Model Evaluation (Linear Regression)")
        
    st.write('R-squared:', round(r2_score_lr, 2))
    st.write('MSE:', round(mse_lr, 2)) 
    st.write('MAE:', round(mae_lr, 2))
    
    st.subheader("Predicted vs Actual Prices (Linear Regression)")
    st.image("linear_regression.png", width=image_width)
    
    st.subheader("Decision Tree Regression")           
    st.write("Fitting the Model")
    st.code("""
            dt_regressor = DecisionTreeRegressor(random_state=1)
            dt_regressor.fit(X_train_new, y_train)
            y_predict_dt = dt_regressor.predict(X_test_new)""", language='python')
    
    st.subheader("Model Evaluation (Decision Tree Regression)")
    st.image("decision_tree_regressor.png", width=image_width)
    
    st.write('R-squared:', round(r2_score_dt, 2))
    st.write('MSE:', round(mse_dt, 2)) 
    st.write('MAE:', round(mae_dt, 2))
            
    st.subheader("Predicted vs Actual Prices (Decision Tree)")

    st.subheader("Random Forrest Regression")
    st.write("Find Hyperparameter using GridsearchCV")
    st.code("""
            optimal_depth = {'max_depth': [17,18,19,20,21]}
            rf_reg = RandomForestRegressor(random_state=42)
            rfreg_grid = GridSearchCV(rf_reg, optimal_depth, cv=5)
            rfreg_grid.fit(X_train, y_train)
            optimal_depth = rfreg_grid.best_params_['max_depth']  # 19 is chosen here
            """, language='python')
    
    st.write("Fitting the Model")
    st.code("""
            rf_reg = RandomForestRegressor(max_depth=optimal_depth, random_state=42)
            rf_reg.fit(X_train, y_train)
            pred_rfreg_train = rf_reg.predict(X_train)
            pred_rfreg_test = rf_reg.predict(X_test)
            """, language='python')
            
    st.subheader("Model Evaluation (Random Forrest Regression)")
    
    
    st.write('R-squared:', round(r2_score_rfreg, 2))
    st.write('MSE:', round(mse_rfreg, 2)) 
    st.write('MAE:', round(mae_rfreg, 2))
    
    st.subheader("Predicted vs Actual Prices (Random Forrest)")
    st.image("random_forrest_regressor.png", width=image_width)
    

elif selected_page == "Car Price Prediction":
    st.title("Car Price Prediction")
    st.write("Enter the details of the car to get a price prediction based of the results of the best performing model: Random Forrest Regression!")

    # Input fields for the user
    mileage = st.number_input("Mileage:", min_value=0, max_value=500000, step=1000, value=235)
    hp = st.number_input("Horsepower:", min_value=50, max_value=1000, step=10, value=116)
    year = st.selectbox("Year:", range(2011, 2022))
    fuel = st.selectbox("Fuel Type:", df['fuel'].unique())
    offertype = st.selectbox("Offer Type:", df['offerType'].unique())
    gear = st.selectbox("Gear Type:", df['gear'].unique())
    brand = st.selectbox("Brand:", df['marke'].unique())
    model = st.selectbox("Model", df['model'].unique())
    #note: This could further be improved by limiting the selection options for the model to models which are exclusive for the brand.


    inputs = {
        "mileage": mileage,
        "marke": brand,
        "model": model,
        "fuel": fuel,
        "gear": gear,
        "offerType": offertype,
        "hp": hp,
        "year": year
    }

    # Create a DataFrame from the user inputs
    input_df = pd.DataFrame(inputs, index=[0])

    # Perform one-hot encoding on categorical variables
    input_encoded = pd.get_dummies(input_df, columns=["marke", "model", "fuel", "gear", "offerType", "year"])

    # Create a DataFrame with all possible dummy variables
    all_vars = pd.DataFrame(columns=feature_names)

    # Concatenate input_encoded with all_vars
    input_final = pd.concat([all_vars, input_encoded], axis=0, ignore_index=True, sort=False).fillna(0)

    # Sort the features in the final input dataframe 
    input_final_sorted = input_final.reindex(columns=feature_names)

    # Make predictions using random forest regression
    

    # Button to trigger prediction
    if st.button("Get Price Prediction"):
        # Prepare input for the model
        
        # Make prediction
        predicted_price = rf_reg_prediction = rf_reg.predict(input_final_sorted)

        # Display the predicted price
        st.success(f"The predicted price for the car is: {round(predicted_price[0]*1000, 2)} €")












































