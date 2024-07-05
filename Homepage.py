import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

df = pd.read_csv('FMCG_data.csv')
df['workers_num'] = df['workers_num'].fillna(df['workers_num'].median())
df.drop(columns=['wh_est_year'] , inplace = True)

st.title('Inventory Management')


   