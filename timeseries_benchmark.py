import pandas as pd
import streamlit as st
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
tqdm.pandas()
from BenchModelClass import BenchModel

st.title('TimeSeries Benchmark')

#######Upload data files#######
st.header('Upload dataset')
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)

#######Choose the time series column#######
time_series_column = st.selectbox(
                                  "Please choose the column that you want to forecast",
                                  dataframe.columns
                                    )
#######Choose darts models#######
