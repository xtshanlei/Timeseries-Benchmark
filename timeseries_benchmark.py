import pandas as pd
import streamlit as st
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
tqdm.pandas()
from BenchModelClass import BenchModel
def mape(y_true,y_pred):
  return '{}%'.format(round((np.mean(np.abs((y_true - y_pred) / y_true)) * 100),2))
#########################################################
st.sidebar.title('TimeSeries Forecasting')

#######Upload data files#######
st.header('1. Upload dataset')
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)

#######Choose the time series column#######
if uploaded_file:
    st.header('2. Which column to forecast?')
    timeseries_to_forecast = st.selectbox(
                                      "Please choose the column that you want to forecast",
                                      dataframe.columns
                                        )
    if timeseries_to_forecast:
        date_column_name = st.selectbox(
                                          "Please choose the column of the date",
                                          dataframe.columns
                                            )
        if date_column_name:

            date_format = st.text_input("Please indicate the date format below (%Y: year; %m: month; %d: day; ")
            #######Choose darts models#######
            st.header('3. Model selection')
            if date_format:
                selected_model_list = st.multiselect(
                                                    "Please choose model that you want to use",
                                                    ['Arima','FFT','FacebookProphet','TCN','LSTM','NBEATS']
                                                    )
                if st.button('Start the training...'):
                    paras ={
                        "dataframe":dataframe,
                        "date_column":date_column_name,
                        "date_format":date_format,
                        "split_point":'2017-03-27',
                        "y_column":timeseries_to_forecast,
                        "num_lags":1,
                        "input_length":30,
                        "kernel_size":12,
                        "model_list": selected_model_list,
                        "scaled": True,
                        }
                    st.write(selected_model_list)
                    forecasting_model = BenchModel(**paras)
                    st.write(forecasting_model)
                    pred_df,metric_df = forecasting_model.bench_compare()
                    st.write(metric_df)
