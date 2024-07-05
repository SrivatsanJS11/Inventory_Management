import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def load_data():
    df = pd.read_csv('FMCG_data.csv')
    return df
def preprocess_data(df):
    df['workers_num'] = df['workers_num'].fillna(df['workers_num'].median())
    df.drop(columns=['wh_est_year'] , inplace = True)

if 'df' not in st.session_state:
    st.session_state.df = load_data()
    st.session_state.df = preprocess_data(st.session_state.df)

st.title('Inventory Management')


   