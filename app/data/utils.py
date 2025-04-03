import yfinance as yf
import os 


def download_data(token):
    data = yf.download(token, interval='1d')
    if 'Ticker' in data.columns.names:
        data = data.droplevel('Ticker', axis=1)
    os.makedirs('data', exist_ok=True)
    data.to_csv(f'data/{token}.csv')
    return data
