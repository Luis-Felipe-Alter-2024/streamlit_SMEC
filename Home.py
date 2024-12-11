import pandas as pd
import streamlit as st
import utilidades as util
from PIL import Image



#Página de presentación
st.set_page_config(
    page_title="SMEC",
    page_icon=":chart_with_upwards_trend:",
    initial_sidebar_state='expanded',
    layout="wide"
 )
#llamamos a las otras páginas
util.generarMenu()


#modificaciones
left_col, center_col, right_col = st.columns([1,4,1],vertical_alignment='center')

with center_col:
    st.title("SMEC - Síndrome Metabólico de Enfermedad Cardiovascular")

    st.write("""
    Determinar si un paciente al cual se le realizan diferentes estudios clínicos para hallar enfermedades como: Hipertensión, Hiperglusemia, Colesterol HDL bajo, Hipertriglidicemia, Trastorno de cintura-altura y poliúrea. Además, se le preguntan datos como: Edad, Género, si fuma y si consume licor.

    Todo esto con la finalidad de diagnosticar si la persona posee un síndrome metabólico asociado a enfermedad cardiovascular (SMEC), a la cual llamaremos enferdedad, una variable categórica que vamos a predecir a través del modelo de Bosques Aleatorios (Random Forest).

    Los datos se encuentran en la carpeta:\n\n https://drive.google.com/drive/folders/1_E-q91yPR_wAi__1blntpTC3kXEpBHIb?usp=sharing
    """)


    image = Image.open("media/imagen_2.jpeg")
    st.image(image, caption="Enfermedad Cardiovascular", use_container_width=True, width=700)

    st.header("Key Performance Indicators (KPIs)")

    st.write("""
    - KPI: Identificar a través de los parámetros de las enfermedades de base de cada paciente, y sus datos médicos generales como el género, la edad, si consume o no, tabaco y alcohol, para determinar si el paciente puede padecer SMEC.

    """)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 




