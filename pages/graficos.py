import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
import utilidades as util
from pickle import dump
from pickle import load
import numpy as np
import plotly.express as px


#Página de presentación
st.set_page_config(
    page_title="SMEC",
    page_icon=":chart_with_upwards_trend:",
    initial_sidebar_state="expanded",
    layout="wide"
 )
#llamamos a las otras páginas
util.generarMenu()

#título
st.title("SMEC - Síndrome Metabólico de Enfermedad Cardiovascular")

df = pd.read_csv("data/Datos_Pacientes.csv", index_col=0)

#selector de gráficos
st.header('Visualizador de Gráficos') 
tipo = st.selectbox('Seleccione el tipo de gráfico', ["Barras", "Líneas", "Dispersión"] )
#selector de variables a comparar
variable = st.selectbox('Seleccione la variable a comparar', df.columns[1:].values)

if tipo == 'Barras':
    fig = px.bar(df, x='Enfermedad', y=f"{variable}", barmode='group', title=f'Pacientes con o sin SMEC que presentan {variable}')
elif tipo == 'Líneas':
    fig = px.line(df, x='Enfermedad', y=f"{variable}", title=f'Pacientes con o sin SMEC que presentan {variable}')
elif tipo == 'Dispersión':
    fig = px.scatter(df, x='Enfermedad', y=f"{variable}", title=f'Pacientes con o sin SMEC que presentan {variable}')

st.plotly_chart(fig)

