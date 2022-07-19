from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import Prophet
from darts.models import NaiveDrift
from darts.models import FFT
from darts.models import ARIMA
from darts.models import RNNModel
from darts.models import ExponentialSmoothing
from darts.models import NBEATSModel
from darts.models import TCNModel
from darts.models import VARIMA
import pandas as pd
import json
import numpy as np
from sklearn.metrics import mean_absolute_error
import streamlit as st
class BenchModel:


  def __init__(self,dataframe,date_column,date_format,split_point,y_column,model_list,num_lags,input_length,kernel_size,seasonal=None,scaled = True):
    self.dataframe = dataframe
    self.date_column = date_column
    self.date_format = date_format
    self.split_point = split_point
    self.scaled = scaled
    self.y_column = y_column
    self.model_list = model_list
    self.seasonal = seasonal
    self.num_lags = num_lags
    self.input_length = input_length
    self.kernel_size = kernel_size
  def mape(self,y_true,y_pred):
    return '{}%'.format(round((np.mean(np.abs((y_true - y_pred) / y_true)) * 100),2))
  def import_dataset(file_url):
    df = pd.read_csv(file_url)
    df = df.drop('Unnamed: 0',axis =1)
    return df
  def prepare(self):
    self.dataframe[self.date_column]=pd.to_datetime(self.dataframe[self.date_column],format=self.date_format)
  def convert_2_timeseries(self):
    self.series = TimeSeries.from_dataframe(self.dataframe,time_col=self.date_column,value_cols=self.y_column)
  def series_train_test_split(self):
    self.train_series, self.val_series = self.series.split_before(pd.Timestamp(self.split_point))
    print("Train size: {}; Test size: {}".format(len(self.train_series),len(self.val_series)))

  def series_scale(self):
    if self.scaled == False:
      self.series_transformed = self.series
      self.train_transformed = self.train_series
      self.val_transformed = self.val_series
    else:
      self.scaler = Scaler()
      self.series_transformed = self.scaler.fit_transform(self.series)
      self.train_transformed = self.scaler.transform(self.train_series)
      self.val_transformed = self.scaler.transform(self.val_series)
  def prophet(self):
    model_prophet = Prophet()
    model_prophet.fit(self.train_transformed)
    if self.scaled == True:
      self.pred_prophet = self.scaler.inverse_transform(model_prophet.predict((len(self.val_series))))
    else:
      self.pred_prophet = model_prophet.predict(len(self.val_series))

  def tcn(self):
    model_tcn = TCNModel(input_chunk_length=self.input_length,output_chunk_length=1,kernel_size=self.kernel_size,batch_size=2,n_epochs=100)
    model_tcn.fit(self.train_transformed)
    if self.scaled == True:
      self.pred_tcn = self.scaler.inverse_transform(model_tcn.predict((len(self.val_series))))
    else:
      self.pred_tcn = model_tcn.predict(len(self.val_series))

  def naive_drift(self):
    model_NDrift = NaiveDrift()
    model_NDrift.fit(self.train_transformed)
    if self.scaled == True:
      self.pred_NDrift = self.scaler.inverse_transform(model_NDrift.predict((len(self.val_series))))
    else:
      self.pred_NDrift = model_NDrift.predict(len(self.val_series))

  def fft(self):
    model_fft = FFT()
    model_fft.fit(self.train_transformed)
    if self.scaled == True:
      self.pred_fft = self.scaler.inverse_transform(model_fft.predict((len(self.val_series))))
    else:
      self.pred_fft = model_fft.predict(len(self.val_series))

  def arima(self):
    model_arima = ARIMA(p=self.num_lags,d=0)
    model_arima.fit(self.train_transformed)
    if self.scaled == True:
      self.pred_arima = self.scaler.inverse_transform(model_arima.predict((len(self.val_series))))
    else:
      self.pred_arima = model_arima.predict(len(self.val_series))
  def lstm(self):
    model_lstm = RNNModel(model='LSTM',input_chunk_length=self.input_length,output_chunk_length=1,batch_size=2,n_epochs=100,training_length=self.kernel_size)
    model_lstm.fit(self.train_transformed)
    if self.scaled == True:
      self.pred_lstm = self.scaler.inverse_transform(model_lstm.predict((len(self.val_series))))
    else:
      self.pred_lstm = model_lstm.predict(len(self.val_series))

  def ex_smoothing(self):
    model_ex_smoothing = ExponentialSmoothing(seasonal =self.seasonal)
    model_ex_smoothing.fit(self.train_transformed)
    if self.scaled == True:
      self.pred_ex_smoothing = self.scaler.inverse_transform(model_ex_smoothing.predict((len(self.val_series))))
    else:
      self.pred_ex_smoothing = model_ex_smoothing.predict(len(self.val_series))

  def varima(self):
    model_varima = VARIMA(p=self.num_lags)
    model_varima.fit(self.train_transformed)
    if self.scaled == True:
      self.pred_varima = self.scaler.inverse_transform(model_varima.predict((len(self.val_series))))
    else:
      self.pred_varima = model_varima.predict(len(self.val_series))

  def nbeats(self):
    model_nbeats = NBEATSModel(input_chunk_length=self.input_length,output_chunk_length=1)
    model_nbeats.fit(self.train_transformed)
    if self.scaled == True:
      self.pred_nbeats = self.scaler.inverse_transform(model_nbeats.predict((len(self.val_series))))
    else:
      self.pred_nbeats = model_nbeats.predict(len(self.val_series))


  def bench_compare(self):
    self.prepare()
    self.convert_2_timeseries()
    self.series_train_test_split()
    self.series_scale()
    self.pred_results = pd.DataFrame()
    self.pred_results['Actual'] = pd.Series([num[0] for num in json.loads(self.val_series.to_json())['data']]).astype(int)
    self.pred_results['date'] = pd.to_datetime(json.loads(self.val_series.to_json())['index'])

    i =0
    training_progress_bar = st.progress(i)
    for model in self.model_list:
      if model =='FacebookProphet':
        st.write('Training using '+model)
        self.prophet()
        self.pred_results[model] = pd.Series([num[0] for num in json.loads(self.pred_prophet.to_json())['data']]).astype(int)
        st.write('Training completed!')

      if model== 'NaiveDrift':
        st.write('Training using '+model)
        self.naive_drift()
        self.pred_results[model] = pd.Series([num[0] for num in json.loads(self.pred_NDrift.to_json())['data']]).astype(int)

      if model== 'FFT':
        st.write('Training using '+model)
        self.fft()
        self.pred_results[model]= pd.Series([num[0] for num in json.loads(self.pred_fft.to_json())['data']]).astype(int)

      if model== 'Arima':
        st.write('Training using '+model)
        self.arima()
        self.pred_results[model]= pd.Series([num[0] for num in json.loads(self.pred_arima.to_json())['data']]).astype(int)

      if model== 'LSTM':
        st.write('Training using '+model)
        self.lstm()
        self.pred_results[model]= pd.Series([num[0] for num in json.loads(self.pred_lstm.to_json())['data']]).astype(int)

      if model== 'ExponentialSmoothing':
        st.write('Training using '+model)
        self.ex_smoothing()
        self.pred_results[model]= pd.Series([num[0] for num in json.loads(self.pred_ex_smoothing.to_json())['data']]).astype(int)

      if model=='NBEATS':
        st.write('Training using '+model)
        self.nbeats()
        self.pred_results[model]= pd.Series([num[0] for num in json.loads(self.pred_nbeats.to_json())['data']]).astype(int)

      if model=='TCN':
        st.write('Training using '+model)
        self.tcn()
        self.pred_results[model]= pd.Series([num[0] for num in json.loads(self.pred_tcn.to_json())['data']]).astype(int)
      if model=='VARIMA':
        st.write('Training using '+model)
        self.varima()
        self.pred_results[model]= pd.Series([num[0] for num in json.loads(self.pred_varima.to_json())['data']]).astype(int)
      i = int(i+1/len(self.model_list)*100)
      st.write(i)
      training_progress_bar.progress(i)

    self.metric_results = pd.DataFrame(columns =['Model','MAE','MAPE'])
    c = 0
    for pred_model_name in self.model_list:
      actual_data = self.pred_results['Actual']
      pred_data = self.pred_results[pred_model_name]
      mae_result = mean_absolute_error(actual_data,pred_data)
      mape_result = self.mape(actual_data,pred_data)
      print(mape_result)
      self.metric_results.loc[c]=[pred_model_name,mae_result,mape_result]
      c+=1
    return self.pred_results,self.metric_results
