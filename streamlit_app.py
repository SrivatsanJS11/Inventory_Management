import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

df = pd.read_csv('FMCG_data.csv')
df['workers_num'] = df['workers_num'].fillna(df['workers_num'].median())
df.drop(columns=['wh_est_year'] , inplace = True)
st.title('Inventory Management')
tabs = st.tabs(['Introduction' , 'EDA' , 'ML'])
with tabs[0]:
    st.markdown("""<h4 style='margin-bottom:0;'>Background</h4>
                <p style='margin-top:0;'>The fast-moving consumer goods (FMCG) industry is highly competitive, and efficient supply chain management is crucial for maintaining profitability and customer satisfaction. Two years ago, a FMCG company ventured into the instant noodles market. Despite the initial success, the higher management has identified a critical issue: a mismatch between supply and demand across different regions. Where demand is high, the supply is insufficient, and where demand is low, there is an oversupply. This discrepancy leads to increased inventory costs and potential loss of sales.</p>"""
                , unsafe_allow_html=True)
    st.markdown("""
<h4 style='margin-bottom:0;'>Problem Statement</h4>
<p style='margin-top:0;'>The primary challenge is to optimize the supply quantities to each warehouse across the country. The goal is to ensure that the supply matches the demand as closely as possible, minimizing inventory costs and maximizing customer satisfaction. Additionally, understanding the demand patterns in various regions will help drive targeted advertising campaigns, further boosting sales in high-demand areas.</p>
""", unsafe_allow_html=True)
    st.markdown("""
<h4 style='margin-bottom:0;'>Objective</h4>
<p style='margin-top:0;'>The objective of this project is twofold:
<ol>
    <li>Optimize Supply Quantities: Build a predictive model using historical data to determine the optimal weight of instant noodles to be shipped to each warehouse. This model will help in aligning the supply with the actual demand, reducing excess inventory costs, and avoiding stockouts.</li>
    <li>Analyze Demand Patterns: Conduct an in-depth analysis of demand patterns across different regions. This analysis will help the management identify high-demand pockets and design targeted advertising campaigns to boost sales in these areas.</li>
</ol>
</p>
""", unsafe_allow_html=True)
    st.markdown('<h4>Dataset</h4>' , unsafe_allow_html=True)
    st.dataframe(df.head(10))

with tabs[1]:
    st.markdown('<h4>Exploratory Data Analysis</h4>', unsafe_allow_html = True)
    fig = px.histogram(
        df,
        x='zone',
        color='WH_regional_zone',
        title='Distribution of Warehouses Across Different Regions and Zones',
        category_orders={"WH_regional_zone": ["Zone 1", "Zone 2","Zone 3","Zone 4","Zone 5","Zone 6"]},
        text_auto=True
    )
    st.plotly_chart(fig)
    st.markdown("<h5>Insight:</h5>" , unsafe_allow_html=True)
    st.markdown("""This stacked bar chart illustrates the distribution of warehouses across different regions (West, South, North, and East) and zones (Zone 1 to Zone 6). 
            The West region has a total of 8,911 warehouses, with Zone 6 having the highest count (2,398) and Zone 1 the lowest (490). The South region has 5,682 warehouses, 
            where Zone 6 also leads (1,364) and Zone 1 is minimal (680). The North region contains 11,282 warehouses, with Zone 6 again dominant (4,519) and Zone 1 the least (841). 
            The East region has the smallest total of 58 warehouses, with each zone contributing similarly. This chart highlights significant regional disparities in warehouse distribution, 
            with the North region having the highest concentration, particularly in Zone 6""")
    fig = px.histogram(df, x='zone', y='Competitor_in_mkt', color='WH_regional_zone', title='Number of Competitors in the Market Across Different Regions and Zones', barmode='group')
    st.plotly_chart(fig)
    st.markdown("<h5>Insight:</h5>" , unsafe_allow_html=True)
    st.markdown("""This bar chart shows the number of competitors in the market across different regions (West, South, North, and East) and zones (Zone 1 to Zone 6). 
                In the West region, Zone 6 has the highest number of competitors at around 9,000, followed by Zone 5 with approximately 5,000. In the South, Zone 4 leads 
                with around 7,000 competitors, while other zones show lower counts. The North region also sees a peak in Zone 6, with approximately 13,000 competitors, 
                followed by Zone 5 with around 5,000. The East region has significantly fewer competitors, with each zone contributing minimally. This chart highlights 
                the competitive landscape, with Zone 6 being particularly prominent in both the West and North regions.""")
    fig = px.scatter(df, x='Competitor_in_mkt', y='product_wg_ton', color='zone', title='Impact of Competitors on Product Demand')
    st.plotly_chart(fig)
    st.markdown("<h5>Insight:</h5>" , unsafe_allow_html=True)
    st.markdown("""This scatter plot demonstrates the impact of competitors on product demand across different regions (West, North, South, and East). 
                The x-axis represents the number of competitors in the market, while the y-axis shows product weight in tons. Each color corresponds to a 
                different region: West (red), North (orange), South (green), and East (purple). The plot reveals a significant clustering of data points 
                between 0 to 8 competitors, with product weight varying from 10,000 to over 50,000 tons. Notably, the South and East regions (green and purple) 
                exhibit the highest product weights across various competitor counts. The East region also shows some outliers with up to 12 competitors. 
                This chart highlights the relationship between market competition and product demand across different regions.""")
    fig = px.histogram(df, x='zone', y='govt_check_l3m', title='Frequency of Government Checks in Different Regions', text_auto=True)
    st.plotly_chart(fig)
    st.markdown("<h5>Insight:</h5>" , unsafe_allow_html=True)
    st.markdown("""The North zone has the most frequent government closures, at around 204,827. The West zone has a mid-range value of government closures, 
                at around 126,761. The South zone has a value close to the West zone, at around 128,944. The East zone has the least frequent government closures, 
                at around 9,775. Overall, the graph suggests a significant difference in the frequency of government closures between the North zone and the other three zones. 
                The West, South, and East zones have comparable closure rates, while the North zone experiences substantially more closures.""")
    
with tabs[2]:
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
