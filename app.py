import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt
import datetime
import torch
from chronos import ChronosPipeline
import numpy as np


def predict_chronos_t5(data, prediction_length):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    pipeline = ChronosPipeline.from_pretrained(
        'amazon/chronos-t5-tiny',
        device_map = device,
        torch_dtype = torch.float32
    )
    context = torch.tensor(data=data['Close'], dtype=torch.float32)

    forecast = pipeline.predict(context, prediction_length, num_samples=100,)
    low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
    return median
# Title and description 
st.title('Crypto Price Prediction 🚀')
st.write('''
This app allows you to upload a BTC-USD dataset, visualize historical data, and predict the price. 
''')

# Upload the dataset
uploaded_file = st.file_uploader('Upload your crypto dataset', type=['csv'])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')
    st.success('Dataset uploaded successfully!')
else:
    st.info('Using BTC-USD as an example')
    data = pd.read_csv('data/BTC-USD.csv', parse_dates=['Date'], index_col='Date')

# Visualize data uploaded

if st.checkbox('Show Data'):
    st.write(data.head())

if st.checkbox('Show Historical Data'):
    st.subheader("Historical Data Visualization")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(data.index, data['Close'],color='black', label="Closing Price", linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title("Historical Data")
    ax.legend()
    st.pyplot(fig)


# Days to forecast 
prediction_length = st.number_input('Days to predict', min_value=1, max_value=30, value=7, step=1)

# Button to make a prediction and show results

if st.button('Make a pediction'):
    st.subheader(f'The price for the following {prediction_length} days is ...')
    prediction = predict_chronos_t5(data,prediction_length)
    st.write(prediction)

