import pandas as pd
import streamlit as st
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
tqdm.pandas()
from BenchModelClass import BenchModel
#########################################################
st.sidebar.title('TimeSeries Forecasting')

#######Upload data files#######
st.header('1. Upload dataset')
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)

#######Choose the time series column#######
st.header('2. Which column to forecast?')
timeseries_to_forecast = st.selectbox(
                                  "Please choose the column that you want to forecast",
                                  dataframe.columns
                                    )
date_column = st.selectbox(
                                  "Please choose the column of the date",
                                  dataframe.columns
                                    )

date_format = st.text_input("Please indicate the date format below (%Y: year; %m: month; %d: day; ")
#######Choose darts models#######
st.header('3. Model selection')
selected_model_list = st.multiselect(
                                    "Please choose model that you want to use",
                                    ['Arima','FFT','FacebookProphet','TCN','LSTM','NBEATS']
                                    )
paras ={
    "dataframe":dataframe,
    "date_column":date_column,
    "date_format":date_format,
    "split_point":'2017-03-27',
    "y_column":timeseries_to_forecast,
    "num_lags":1,
    "input_length":30,
    "kernel_size":12,
    "model_list": ['Arima','FFT','FacebookProphet','TCN','LSTM','NBEATS'],
    "scaled": True,
    }
london_model = BenchModel(**paras)
pred_london_df,metric_london_df = london_model.bench_compare()
