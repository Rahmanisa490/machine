import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor as KNN_Reg
from sklearn.metrics import mean_squared_error as mse

# Load data
data_location = 'path_to_your/toyota.csv'
row_data = pd.read_csv(data_location)

# Drop missing values
data = row_data.dropna(axis=0)

# Define features and target
features = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
x = data[features]
y = data['price']

# Split data into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=70)

# Define and train the model
model = KNN_Reg(n_neighbors=80)
model.fit(train_x, train_y)
acc1 = model.score(test_x, test_y)
test_predict = model.predict(test_x)
score = mse(test_predict, test_y)

# New model after tuning
new_model = KNN_Reg(n_neighbors=97)
new_model.fit(train_x, train_y)
acc2 = new_model.score(test_x, test_y)

# Streamlit UI
st.title('Used Car Price Prediction')
st.write('This app predicts the price of a used car based on its features.')

# Input data
year = st.number_input('Year', min_value=1990, max_value=2022, value=2019)
mileage = st.number_input('Mileage', min_value=0, max_value=200000, value=5000)
tax = st.number_input('Tax', min_value=0, max_value=600, value=145)
mpg = st.number_input('MPG', min_value=0.0, max_value=200.0, value=30.2)
engine_size = st.number_input('Engine Size', min_value=0.0, max_value=10.0, value=2.0)

# Predict button
if st.button('Predict'):
    input_data = np.array([[year, mileage, tax, mpg, engine_size]])
    prediction_old = model.predict(input_data)
    prediction_new = new_model.predict(input_data)
    st.write(f'Prediction with old model: £{prediction_old[0]:.2f}')
    st.write(f'Prediction with new model: £{prediction_new[0]:.2f}')
    st.write(f'Prediction in Rupiah with old model: Rp{prediction_old[0] * 19110:.2f}')
    st.write(f'Prediction in Rupiah with new model: Rp{prediction_new[0] * 19110:.2f}')

# Display accuracy improvement
st.write(f'Old Model Accuracy: {acc1:.2f}')
st.write(f'New Model Accuracy: {acc2:.2f}')
st.write(f'Improvement: {acc2 - acc1:.2f}')
