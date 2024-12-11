import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
import utilidades as util
from pickle import dump
from pickle import load
import numpy as np


#Configuramos la página
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

#modificación esto lo llevamos a utilidades dentro de la función

#st.markdown('**Datos de enferemedades de pacientes - Construcción propia.')
#st.write(df, unsafe_allow_html=False)    
#st.subheader('Resultado del modelo Random Forest para los datos históricos')

util.modelo_rf(df)

#Ingresar datos a través de entradas para cada variable

with st.sidebar:
    st.header('Datos para el Diagnóstico')
    col1, col2 =st.columns(2)
    
    with col1:
        
        hipertension = st.checkbox('Hipertensión')
        hiperglucemia = st.checkbox('Hiperglucemia')
        hdl = st.checkbox('HDL Bajo')
        hipertri = st.checkbox('Hipertri/cemia')
        
    with col2:
        ica = st.checkbox('ICA')
        tabaco = st.checkbox('Tabaco')
        alcohol = st.checkbox('Alcohol')
        poliu = st.checkbox('Poliúrea')
    
    edad = st.number_input('Edad', min_value=18)
    genero = st.selectbox('Género',('Femenino','Masculino'))

    col3, col4 =st.columns(2)
    with col4:
        btn_ejecutar = st.button('Diagnosticar',type='primary')





lista = [0,0,0,0,0,0,0,0,0,0]

if btn_ejecutar == True:
    if hipertension == True:
        lista[0] = 1
    else:
        lista[0] = 0
    if hiperglucemia == True:
        lista[1] = 1
    else:
        lista[1] = 0
    if hdl == True:
        lista[2] = 1
    else:
        lista[2] = 0
    if hipertri == True:
        lista[3] = 1
    else:
        lista[3] = 0
    if ica == True:
        lista[4] = 1
    else:
        lista[4] = 0
    lista[5]= edad

    if genero == 'Masculino':
        lista[6] = 1
    else:
        lista[6] = 0
    if tabaco == True:
        lista[7] = 1
    else:
        lista[7] = 0
    if alcohol == True:
        lista[8] = 1
    else:
        lista[8] = 0
    if poliu == True:
        lista[9] = 1
    else:
        lista[9] = 0
    arr = np.array([lista])    
    util.prueba_del_modelo(arr, df)
    
