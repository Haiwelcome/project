import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

model = joblib.load("risk_prediction_model.pkl")
risk_encoder = joblib.load("risk_label_encoder.pkl")

df = pd.read_csv("final_dashboard_dataset.csv")

state = st.sidebar.selectbox("Select State", df['state'].unique())
district = st.sidebar.selectbox("Select District", df[df['state']==state]['district'].unique())

df_filtered = df[(df['state']==state) & (df['district']==district)].copy()

df_filtered['scheme_type_enc'] = df_filtered['scheme_name'].astype('category').cat.codes

feature_cols = ['gap_percent','missed_population','population','coverage_percent_calc','scheme_type_enc']
df_filtered['predicted_risk'] = model.predict(df_filtered[feature_cols])
df_filtered['predicted_risk'] = risk_encoder.inverse_transform(df_filtered['predicted_risk'])

st.title("Prevention Gap Radar – AI Dashboard")
st.dataframe(df_filtered[['area_name','scheme_name','predicted_risk','gap_category','priority_score']])

fig = px.scatter_mapbox(
    df_filtered,
    lat="latitude",
    lon="longitude",
    color="predicted_risk",
    hover_name="area_name",
    zoom=6,
    mapbox_style="carto-positron"
)
st.plotly_chart(fig, use_container_width=True)
