import pandas as pd
import streamlit as st
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
tqdm.pandas()
import h2o
from h2o.automl import H2OAutoML
from BenchModelClass import BenchModel
from

#######Upload data files#######
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)
