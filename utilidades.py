import streamlit as st
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import utilidades as util
from pickle import dump
from pickle import load
import numpy as np
import plotly.express as px
from sklearn import tree
from sklearn.tree import export_graphviz
import graphviz
from matplotlib import pyplot as plt
from sklearn import metrics
from PIL import Image

def generarMenu():
    with st.sidebar:
        col1, col2 = st.columns(2)
        with col1:
            image = Image.open("media\icono_pag2.png")
            st.image(image, use_container_width=False, width=80 )
        with col2:
            st.header('SMEC')

        st.page_link('Home.py', label='Inicio', icon='🏠')
        st.page_link('pages/pronostico.py', label='Pronóstico SMEC', icon='💚')
        st.page_link('pages/arboles.py', label='Árboles', icon='🌳')
        st.page_link('pages/graficos.py', label='Gráficos', icon='📊')


#Función del modelo predictivo
def modelo_rf(df_pacientes_rf):
    left_col, center_col, right_col = st.columns([0.5,5,0.5], vertical_alignment='center')

    with center_col:
        #subtítulo
        #st.header("Datos")
        st.markdown('**Datos de enferemedades de pacientes - Construcción propia.')
        st.write(df_pacientes_rf, unsafe_allow_html=False)    
        st.subheader('Resultado del modelo Random Forest para los datos históricos')
        #Variable a predeci
        Y = df_pacientes_rf.iloc[:,0]
        #Variables predictoras
        X = df_pacientes_rf.iloc[:,1:11] 
        #Variables de prueba  ->  prueba
        #Variables de entrenamiento -> entrenar
        X_entrenar, X_prueba, Y_entrenar, Y_prueba = train_test_split(X, Y, train_size=0.8, random_state=42)
        
        st.markdown('### - *Separamos los datos*')
        st.write('Datos de entrenamiento')
        #st.info(X_entrenar.shape)
        #modificacion
        st.info(f'Muestra de las variables predictoras: {X_entrenar.shape[0]} datos')
        st.info(f'Muestra de la variable a predecir: {X_entrenar.shape[1]} datos')

        st.write('Datos de prueba')
        #st.info(X_prueba.shape)
        #modificacion
        st.info(f'Muestra de las variables predictoras: {X_prueba.shape[0]} datos')
        st.info(f'Muestra de la variable a predecir: {X_prueba.shape[1]} datos')

        st.markdown('### - *Detalle de las variables*')
        #SE modificó para tener mejor vista
        st.write('Variables Predictoras')
        lista = list(X.columns)  
        delim = ", "
        text = delim.join(lista)
        st.info(text)

        st.write('Variable a Predecir')
        st.info(Y.name)
        
        
        #Creamos el bosque
        bosque = RandomForestClassifier()
        #entrenamos el bosque
        bosque.fit(X_entrenar,Y_entrenar)
    

        st.markdown('### - *Desempeño del modelo Random Forest*')
        #Hacemos la predicción
        Y_prediccion = bosque.predict(X_prueba)
        accuracy = accuracy_score(Y_prueba,Y_prediccion)
        st.write('Métrica de precisión de puntos obtenidos - Accuracy:')
        st.info(accuracy)

        #matriz de confusión
        matriz_conf = confusion_matrix(Y_prueba, Y_prediccion)
        st.write('Métrica de la matriz de confusión:')
        #st.info(matriz_conf)
    

        fig, ax = plt.subplots(figsize=(3,3))
        ax.imshow(matriz_conf)
        ax.set_yticks([])
        ax.set_xticks([])
        #ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
        #ax.xaxis.set_minor_formatter(plt.FormatStrFormatter('%.1f'))
        #ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
        #ax.yaxis.set_minor_formatter(plt.FormatStrFormatter('%.1f'))
        ax.text(-0.1,0.05,matriz_conf[0,0], color='black', fontweight='bold')
        ax.text(-0.1,1.05,matriz_conf[0,1], color='white', fontweight='bold')
        ax.text(1,0.05,matriz_conf[1,0], color='white', fontweight='bold')
        ax.text(1,1.05,matriz_conf[1,1], color='white', fontweight='bold') 
        #valores
        ax.text(-0.45,0.2,'Predicción: "SI"', color='black')  
        ax.text(0.5,1.2,'Predicción: "NO"', color='white')  
        st.pyplot(fig, use_container_width=False )

    #Guardamos el modelo
    archivo_modelo = open('data\modelo_rf.sav', 'wb')
    dump(bosque, archivo_modelo)
    archivo_modelo.close()
    return bosque

#Ejecutar el modelo para los nuevos datos

def prueba_del_modelo(arreglo, df_prueba):
    left_col, center_col, right_col = st.columns([0.5,5,0.5], vertical_alignment='center')
    with center_col:
        modelo_cargado = load(open('data\modelo_rf.sav', 'rb'))
        prediccion_rf = modelo_cargado.predict(arreglo)
        st.subheader('Diagnóstico del paciente ingresado')
        st.write('Datos ingresados')

        #Creación del dataframe
        columnas = list(df_prueba.columns[1:].values)
        datos = pd.DataFrame(arreglo, columns=columnas)    
        datos.set_index(datos.columns[0], inplace=True)    
        
        st.write(datos)
        #st.write('Diagnóstico')
        #st.subheader(f'- La persona evaluada de acuerdo a los datos hallados {prediccion_rf[0]} padece de SMEC')
        st.info(f'### La persona evaluada, de acuerdo con los datos ingresados en el modelo,\n ### {prediccion_rf[0]} padece del Síndrome Metabólico de Enfermedad Cardiovascular - SMEC')

def arboles(df_pacientes_rf, num):
    st.write('**Importante: Los árboles pueden variar, ya que cada ejección permite una predicción diferente.')
    #Variable a predeci
    Y = df_pacientes_rf.iloc[:,0]
    #Variables predictoras
    X = df_pacientes_rf.iloc[:,1:11] 
    #Variables de prueba  ->  prueba
    #Variables de entrenamiento -> entrenar
    X_entrenar, X_prueba, Y_entrenar, Y_prueba = train_test_split(X, Y, train_size=0.8, random_state=0)
    #Creamos el bosque
    bosque = RandomForestClassifier(n_estimators=100,   #100 es el valor standart de los árboles del bosque
                                criterion='gini',   #evaluamos la raíz de los árboles a travez de la impureza del gini
                                max_features='sqrt',  #que seleccione la raíz de la cantidad de características o variables predictoras
                                bootstrap=True,      #Seleccionamos que implemente el muestreo por reemplazo
                                max_samples=2/3,     #Cantidad de datos a muestrear
                                oob_score=True)
    #entrenamos el bosque
    bosque.fit(X_entrenar,Y_entrenar)
    #Hacemos la predicción
    Y_prediccion = bosque.predict(X_prueba) 
    dot_data = export_graphviz(bosque.estimators_[num], feature_names=X_entrenar.columns, filled=True, max_depth=5, impurity=True, proportion=True, special_characters=True, rounded=False)
    graph = graphviz.Source(dot_data)
    st.graphviz_chart(graph, use_container_width=False)
    
    
