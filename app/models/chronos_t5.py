from chronos import ChronosPipeline
import torch
import pandas as pd 
import numpy as np



def predict_chronos_t5(data, prediction_length=7, model='tiny'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    pipeline = ChronosPipeline.from_pretrained(
        f'amazon/chronos-t5-{model}',
        device_map = device,
        torch_dtype = torch.float32
    )
    context = torch.tensor(data=data['Close'].values, dtype=torch.float32)

    forecast = pipeline.predict(context, prediction_length, num_samples=100,)
    low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
    
    #Add dates for the prediction horizon
    last_date = data.index[-1]
    pred_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_length)

    #Create dataframe 
    prediction_df = pd.Series(data=median, index=pred_dates, name='Predictions')
    return prediction_df