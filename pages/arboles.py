import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
import utilidades as util
from matplotlib import pyplot as plt
from sklearn import tree
from PIL import Image




#Página de presentación
st.set_page_config(
    page_title="SMEC",
    page_icon=":chart_with_upwards_trend:",
    initial_sidebar_state="expanded",
    layout="wide"
 )
#llamamos a las otras páginas
util.generarMenu()




#Mostrar los árboles 

with st.sidebar:
    num_arbol = st.number_input('Número del árbol', min_value=0, max_value=100)
    btn_graficar = st.button('Graficar')

#gráfico

df2 = pd.read_csv("data/Datos_Pacientes.csv", index_col=0)
if btn_graficar == True:
    st.header(f'Árbol de Decisión {num_arbol}')
    resultado = util.arboles(df2, num_arbol)
else:
    st.header(f'Árbol de Decisión')
    image = Image.open('media\\arbol_default.png')
    st.image(image, caption="Árboles de Decisión", use_container_width=True)

