import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.markdown("<h4>Feature Engineering</h4>",unsafe_allow_html=True)
st.markdown("<p>to accurately predict the number of SKUs (Stock Keeping Units) for each warehouse, it was necessary to create a new feature called num_SKUs in our dataset. This feature engineering step involves several transformations and calculations to ensure the data is suitable for model training and prediction. Below is the process used to generate the num_SKUs feature:</P>",unsafe_allow_html=True)
code = """
size_mapping = {'Small': 1, 'Medium': 2, 'Large': 3}
df['WH_capacity_size'] = df['WH_capacity_size'].map(size_mapping)
df['WH_capacity_size'] = pd.to_numeric(df['WH_capacity_size'], errors='coerce')
df['WH_capacity_size'].fillna(0, inplace=True)
df['num_SKUs'] = df.apply(lambda row: row['product_wg_ton'] / 100, axis=1)
"""
st.code(code , language='python')
st.markdown("<h4>Linear Regression</h4>",unsafe_allow_html=True)
code = """
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
# Train the regression model
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_reg = reg_model.predict(X_test)
mse_reg = mean_squared_error(y_test, y_pred_reg)
rmse_reg = np.sqrt(mse_reg)
r2_reg = r2_score(y_test, y_pred_reg)
print('Regression Model MSE:', mse_reg)
print('Regression Model RMSE' ,rmse_reg)
print('R2 Score:', r2_reg)
"""
st.code(code,language='python')
st.markdown("""<h5 style= 'margin-bottom:0;'>Output:</h5>""",unsafe_allow_html=True)
st.markdown("""
<p style='margin-bottom:0'>Regression Model MSE: 9.34015855620492e-27</P>
<p style ='margin-bottom:0' >Regression Model RMSE 9.664449573672016e-14</p>
<p>R2 Score: 1.0</p>""",unsafe_allow_html=True)
