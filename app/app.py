import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt
import torch
from chronos import ChronosPipeline
import numpy as np
import yfinance as yf
from models.chronos_t5 import predict_chronos_t5
from data.utils import download_data

# st.set_page_config(layout="wide")  # Configurar página para usar todo el ancho

# Title and description 
st.title('Crypto Price Prediction 🚀')
st.write('''
This app allows to visualize historical data, and predict the price. 
''')

# Sidebar
st.sidebar.title('Data extraction')

token_options = ['ETH-USD','BTC-USD','AAPL-USD']
model_options = ['tiny','base','large']

# Upload the dataset
#uploaded_file = st.sidebar.file_uploader('Upload your crypto dataset', type=['csv'])
selected_option = st.sidebar.selectbox('Choose your favorite token:',token_options, index=None, placeholder='Choose an option')
if st.sidebar.button('Download Update Data',type='primary') and selected_option is not None:
    with st.spinner('Downloading Data...'): 
        data = download_data(selected_option)
        st.session_state.data = data
        st.sidebar.success('Data Downloaded')
elif 'data' in st.session_state:
    st.sidebar.success('Data Downloaded')
else:    
    st.error('Download Data')

# Visualize data uploaded 
# Cheack if data if avaible first 
if 'data' in st.session_state:
    if st.sidebar.checkbox('Show Data'):
        st.sidebar.write(st.session_state.data.head())

    if st.checkbox('Show Historical Data'):
        st.subheader("Historical Data Visualization")
        st.line_chart(st.session_state.data['Close'])
        # fig, ax = plt.subplots(figsize=(15,5))
        # ax.plot(st.session_state.data.index, st.session_state.data['Close'], color='Blue', label="Closing Price", linewidth=0.8)
        # ax.set_xlabel("Date")
        # ax.set_ylabel("Price")
        # ax.set_title("Historical Data")
        # ax.legend()
        # st.pyplot(fig)
    
    col1, col2 = st.columns(2)

    # Days to forecast 
    with col1:
        prediction_length = st.number_input('Days to predict', min_value=1, max_value=30, value=7, step=1)
    with col2:
        model = st.radio('Choose the model:',model_options)



    # Button to make a prediction and show results
    if st.button('Make a prediction', type='secondary'):
        with st.spinner('Thinking...'):
            prediction_df = predict_chronos_t5(st.session_state.data, prediction_length, model)
            st.session_state.prediction_df = prediction_df

    if 'prediction_df' in st.session_state:
        #Title
        st.subheader(f'The price for the following {prediction_length} days is ...')
        # Box with predictions
        if st.toggle('See predictions'):
            st.sidebar.write(st.session_state.prediction_df)
        # Forecasting Plot
        fig2, ax2 = plt.subplots(figsize=(15,5))
        ax2.plot(st.session_state.prediction_df.index, st.session_state.prediction_df, color='Blue', marker='o', label='Forecast')
        ax2.set_xlabel('Dates')
        ax2.tick_params(axis='x', labelrotation=45)
        ax2.set_ylabel('Price')
        ax2.set_title('Forecasted Data')
        ax2.legend()
        st.pyplot(fig2)