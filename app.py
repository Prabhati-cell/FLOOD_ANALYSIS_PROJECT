import streamlit as st
import pandas as pd
import numpy as np
import joblib
from data_processing import load_data, basic_clean, parse_dates
from modeling import prepare_features

st.set_page_config(page_title='Flood Risk Dashboard', layout='wide')

@st.cache_data
def load_cleaned():
    df = load_data()
    df = basic_clean(df)
    return df

df = load_cleaned()

st.title('Climate-Based Flood Risk Analysis and Prediction')

st.sidebar.header('Filters')
state = st.sidebar.selectbox('State', options=['All'] + sorted(df['State'].dropna().unique().tolist())) if 'State' in df.columns else None

if state and state != 'All':
    df = df[df['State'] == state]

st.header('Dataset sample')
st.dataframe(df.head())

st.header('Quick KPIs')
col1, col2, col3 = st.columns(3)
col1.metric('Events', len(df))
if 'Human fatality' in df.columns:
    col2.metric('Total Fatalities', int(pd.to_numeric(df['Human fatality'], errors='coerce').sum(skipna=True)))
if 'Duration(Days)' in df.columns:
    col3.metric('Avg Duration (days)', float(pd.to_numeric(df['Duration(Days)'], errors='coerce').mean()))

st.header('Prediction')
st.write('Load model and make a demo prediction (requires trained model).')
try:
    model = joblib.load('best_model.joblib')
    st.success('Model loaded')
    # simple form: select state, month
    with st.form('predict'):
        st_state = st.selectbox('State', options=['missing'] + sorted(df['State'].dropna().unique().tolist())) if 'State' in df.columns else st.text_input('State')
        st_month = st.slider('Month', 1, 12, 7)
        submitted = st.form_submit_button('Predict')
    if submitted:
        X_input = pd.DataFrame([{'State': st_state, 'month': st_month}])
        pred_prob = model.predict_proba(X_input)[:,1][0]
        pred = model.predict(X_input)[0]
        st.write('Predicted risk (binary):', int(pred))
        st.write('Predicted probability:', float(pred_prob))
except Exception as e:
    st.warning('Model not found or failed to load. Run modeling.py to train and save a model.')
