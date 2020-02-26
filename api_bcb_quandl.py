#import basicos
import quandl
import warnings
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import streamlit as st
from pandas import plotting
import time
from scipy import stats
import statsmodels.api as sm
from fbprophet import Prophet

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf

# for interactive visualizations
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected = True)
import plotly.figure_factory as ff

#filtro de warnings
#warnings.filterwarnings("ignore")

st.title("API BCB QUANDL")

st.header("Escolha um banco de dados")


#quandl BCB params
q_key="" #Necessario adicionar key obtida no Quandl
simb_TS_ipca = "433"
simb_TS_imp = "3034"
simb_TS_exp = "2946"
start_date = "1999-01-01"

def get_TS(simb):
    df = pdr.quandl.QuandlReader('BCB/'+simb,start_date,api_key=q_key).read()
    return df


def get_df(name):
    if name == "ipca":
        df = get_TS(simb_TS_ipca)
        return df
    elif name == "exp":
        df = get_TS(simb_TS_exp)
        return df
    elif name == "imp":
        df = get_TS(simb_TS_imp)
        return df
    else:
        st.warning("Erro nome dataframe")

#df_q variavel para o dataframe a ser tratado
name = st.radio("Qual dataframe deseja",("ipca", "exp","imp"))
df_q = get_df(name)
if st.checkbox("Mostrar dataframe"):
    st.write(df_q)


st.header("Salvar um dataframe")
if st.checkbox("salvar dataframe em csv"):
    nome_csv = st.text_input("nome do csv", "")
    st.text(nome_csv)
    if (nome_csv != "" and len(nome_csv)<20):
        if st.button("salvar dataframe"):
            nome_csv = nome_csv +".csv"
            df_q.to_csv(nome_csv,index=False)
            df_q1 = pd.read_csv(nome_csv)
            st.write(df_q1)

st.header("Analise exploratoria")
if st.checkbox("descrição dataframe: "+name):
    st.write(df_q.describe())
    st.write(df_q.dtypes)

st.header("Graficos")
st.line_chart(df_q)

import logging
logging.getLogger().setLevel(logging.ERROR)

st.header("Using Prophet: ")
df_prophet = df_q
df_prophet.reset_index(level=0, inplace=True)
df_prophet.rename(columns={"Date": 'ds',"Value": "y"}, inplace=True)
if st.checkbox("tipos dados"):
    st.write(df_prophet.dtypes)
df_prophet['ds'] = pd.DatetimeIndex(df_prophet['ds'])
if st.checkbox("tratamento coluna data"):
    st.write(df_prophet.head())
my_model = Prophet()
my_model.fit(df_prophet)
future_dates = my_model.make_future_dataframe(periods=10, freq='MS')
if st.checkbox("datas criadas"):
    st.write(future_dates.tail())
forecast = my_model.predict(future_dates)
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

st.header("Graficos de Previsão - Prophet")
df_for = forecast
df_for = df_for.set_index('ds')
if st.checkbox("tratamento index data"):
    st.write(df_for.tail())
cenario = st.radio("Qual cenário deseja ver", ("medio","pior","melhor"))
if  cenario == "medio":
    st.line_chart(df_for['yhat'])
elif  cenario == "pior":
    st.line_chart(df_for['yhat_lower'])
elif  cenario == "melhor":
    st.line_chart(df_for['yhat_upper'])
else:
        st.warning("Erro de nome")

