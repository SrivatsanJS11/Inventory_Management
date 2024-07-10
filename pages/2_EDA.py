import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

df = pd.read_csv('FMCG_data.csv')
df['workers_num'] = df['workers_num'].fillna(df['workers_num'].median())
df.drop(columns=['wh_est_year'] , inplace = True)

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
fig = px.scatter(df, x='govt_check_l3m', y='storage_issue_reported_l3m', color='zone', title='Correlation between Government Checks and Reported Storage Issues')
st.plotly_chart(fig)
st.markdown("<h5>Insight:</h5>" , unsafe_allow_html=True)
st.markdown("<p>The scatter plot depicts the correlation between government checks conducted in the last three months (govt_check_l3m) and reported storage issues (storage_issue_reported_l3m) across different zones. The zones are color-coded: West (blue), North (red), South (green), and East (purple). Each dot represents a warehouse, showing the frequency of government checks on the x-axis and the number of storage issues on the y-axis. There is no apparent linear trend, suggesting that frequent government checks do not necessarily correlate with the number of reported storage issues. Notably, warehouses in the South zone (green) exhibit a higher concentration of both government checks and reported storage issues.</p>",unsafe_allow_html=True)
