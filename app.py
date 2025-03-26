import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt
import datetime



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