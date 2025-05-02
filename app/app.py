import streamlit as st
import pickle
import pandas as pd

# Cargar el modelo
model = pickle.load(open('/workspaces/detecta_diabetes/app/models/diabetes_histboost_model.pkl', 'rb'))
class_dic = {0: 'diabetes', 1: 'no tiene diabetes'}

st.set_page_config(page_title="DetectaDiabetes", layout="centered")
st.markdown(
    "<description>Esta aplicación predice el riesgo de diabetes basado en indicadores de salud y estilo de vida del BRFSS 2015. "
    "Por favor, ingresa la información solicitada para obtener una predicción.</description>",
    unsafe_allow_html=True
)

# --- Sidebar para navegación ---
page = st.sidebar.radio("Detecta Diabetes", ["Home","Fuente de Datos","Modelos","Modelo seleccionado",
                                             "Importancia de las variables","Predicción"])

if page == "Predicción":
    st.title("🩺 DetectaDiabetes - Predicción de Diabetes")

    # Opciones para los campos
    highbp_options = {"No": "no", "Sí": "yes"}
    highchol_options = {"No": "no", "Sí": "yes"}
    smoker_options = {"No": "no", "Sí": "yes"}
    physactivity_options = {"No": "no", "Sí": "yes"}
    fruits_options = {"No": "no", "Sí": "yes"}
    veggies_options = {"No": "no", "Sí": "yes"}
    hvyalcoholconsump_options = {"No": "no", "Sí": "yes"}
    nodocbccost_options = {"No": "no", "Sí": "yes"}
    diffwalk_options = {"No": "no", "Sí": "yes"}
    sex_options = {"Femenino": "female", "Masculino": "male"}
    age_options = {
        "35-44 años": "entre_35_y_44",
        "45-54 años": "entre_45_y_54",
        "55-64 años": "entre_55_y_64"
    }

    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            highbp = st.selectbox("¿Presión arterial alta?", list(highbp_options.keys()))
            highchol = st.selectbox("¿Colesterol alto?", list(highchol_options.keys()))
            smoker = st.selectbox("¿Fumador/a?", list(smoker_options.keys()))
            physactivity = st.selectbox("¿Actividad física regular?", list(physactivity_options.keys()))
            fruits = st.selectbox("¿Consume frutas diariamente?", list(fruits_options.keys()))
            veggies = st.selectbox("¿Consume vegetales diariamente?", list(veggies_options.keys()))
        with col2:
            hvyalcoholconsump = st.selectbox("¿Consumo excesivo de alcohol?", list(hvyalcoholconsump_options.keys()))
            nodocbccost = st.selectbox("¿No pudo ver al médico por costo?", list(nodocbccost_options.keys()))
            diffwalk = st.selectbox("¿Dificultad para caminar/escaleras?", list(diffwalk_options.keys()))
            sex = st.selectbox("Sexo", list(sex_options.keys()))
            age = st.selectbox("Edad", list(age_options.keys()))
            bmi = st.number_input("Índice de Masa Corporal (BMI)", min_value=10.0, max_value=50.0, value=25.0)
        genhlth = st.slider("Salud general (1=Excelente, 5=Mala)", min_value=1, max_value=5, value=3)
        menthlth = st.slider("Días de mala salud mental (últimos 30 días)", min_value=0, max_value=30, value=0)
        physhlth = st.slider("Días de mala salud física (últimos 30 días)", min_value=0, max_value=30, value=0)
        education = st.slider(
            "Nivel educativo(1= Básico, 2,3 = No se graduó de la preparatoria, 4= Graduado de la preparatoria, 5= Asistió a la universidad o escuela técnica, 6= Graduado de la universidad o escuela técnica)",
                                    min_value=1, max_value=6, value=4)
        income = st.slider("Nivel de ingresos (1=<10k, 2=<15k, 3=<20k, 4=<25k, 5=<35k, 6=<50k, 7=<60k+, 8=75k+)", min_value=1, max_value=8, value=5)

        submitted = st.form_submit_button("Predecir")

    if submitted:
        try:
            input_dict = {
                "highbp": highbp_options[highbp],
                "highchol": highchol_options[highchol],
                "smoker": smoker_options[smoker],
                "physactivity": physactivity_options[physactivity],
                "fruits": fruits_options[fruits],
                "veggies": veggies_options[veggies],
                "hvyalcoholconsump": hvyalcoholconsump_options[hvyalcoholconsump],
                "nodocbccost": nodocbccost_options[nodocbccost],
                "diffwalk": diffwalk_options[diffwalk],
                "sex": sex_options[sex],
                "age": age_options[age],
                "bmi": bmi,
                "genhlth": genhlth,
                "menthlth": menthlth,
                "physhlth": physhlth,
                "education": education,
                "income": income
            }
            input_df = pd.DataFrame([input_dict])

            prediction = int(model.predict(input_df)[0])
            pred_class = class_dic[prediction]

            st.success(f"La predicción es: {pred_class}")
        except Exception as e:
            st.error(f"Error durante la predicción: {e}")

        st.write("**Nota:** Esta predicción es solo orientativa y no sustituye el diagnóstico médico profesional.")

elif page == "Home": 
    st.image("/workspaces/detecta_diabetes/images/presentacion.png")

elif page == "Fuente de Datos":
    st.title("Fuente de Datos") 
    st.markdown("La base de datos utilizada es el BRFSS 2015, que incluye información sobre salud y estilo de vida. " \
    "El Sistema de Vigilancia de Factores de Riesgo Conductual (BRFSS) es una encuesta telefónica relacionada " \
    "con la salud que es recopilada anualmente por los CDC. Cada año, la encuesta recoge respuestas de más de " \
    "400.000 estadounidenses sobre conductas de riesgo relacionadas con la salud, condiciones crónicas de salud" \
    " y uso de servicios preventivos. Se ha llevado a cabo todos los años desde 1984. Para este proyecto, se utilizó " \
    "un csv del conjunto de datos disponible en Kaggle para el año 2015. Este conjunto de datos original contiene " \
    "respuestas de 441.455 individuos y tiene 330 características. Estas características son preguntas directamente " \
    "hechas a los participantes, o variables calculadas basadas en respuestas individuales de los participantes.")   

elif page == "Modelos":
    st.title("Comparación de Modelos")
    st.markdown("A continuación se muestran las métricas de desempeño de los modelos predictores de diabetes:")

    # Mostrar métricas de modelos si existe el archivo
    try:
        metricas_df = pd.read_csv('/workspaces/detecta_diabetes/notebooks/metricas_modelos.csv')
        st.subheader("Métricas de los Modelos")
        # Mejorar visualización: ancho completo, formato, scroll y estilos
        st.dataframe(
            metricas_df.style.format(precision=2).highlight_max(axis=0, color='blue', props='color: white; background-color: blue;'),   
            use_container_width=True,
            hide_index=True
        )
    except Exception as e:
        st.warning(f"No se pudo cargar el archivo de métricas: {e}")
elif page == "Modelo seleccionado":
    st.title("Matriz de Confusión")
    st.markdown("A continuación se muestra la matriz de confusión del modelo HistBoost:")

    # Mostrar matriz de confusión si existe el archivo
    st.image("/workspaces/detecta_diabetes/images/modelo_final.png")

elif page == "Importancia de las variables":
    st.title("Importancia de las Variables")
    st.markdown("A continuación se muestra la importancia de las variables del modelo HistBoost:")

    # Mostrar gráfico de importancia de variables si existe el archivo
    st.image("/workspaces/detecta_diabetes/images/importancia.png")
    st.markdown("El análisis identifica las cinco variables más influyentes en el modelo, proporcionando " \
    "información clave sobre los factores que pueden estar relacionados con la diabetes:"
    "\n\n" \
    "**Genhlth (Salud General)**: La variable con mayor peso en el análisis, lo que sugiere que la percepción" \
    " general de la salud podría ser un fuerte predictor de diabetes. " \
    "\n\n" \
    "**Age (Edad)**: La edad juega un papel crucial en la diabetes, ya que el riesgo de desarrollar la enfermedad " \
    "aumenta significativamente con el envejecimiento. " \
    "\n\n" \
    "**Physhlth (Días de problemas físicos en el último mes)**: Este indicador refleja el impacto físico de la" \
    " enfermedad en la calidad de vida del paciente. Un mayor número de días con problemas físicos podría" \
    " correlacionarse con complicaciones derivadas de la diabetes. " \
    "\n\n" \
    "**HvyAlcoholConsump (Consumo excesivo de alcohol)**: Aunque el consumo de alcohol no siempre está directamente" \
    " asociado con la diabetes, su abuso puede influir negativamente en el metabolismo y la resistencia a la insulina." \
    " También puede contribuir al aumento de peso, otro factor de riesgo clave. " \
    "\n\n" \
    "**BMI (Índice de Masa Corporal)**: La obesidad es uno de los principales factores de riesgo para el desarrollo" \
    " de diabetes tipo 2. Un IMC elevado suele estar asociado con resistencia a la insulina y problemas metabólicos.")     

elif page == "Referencias":
    st.title("Referencias")
    st.markdown("A continuación se presentan las referencias utilizadas en la aplicación:")
    st.markdown(
        """
        1. Diabetes World Health Organization. (9 de Abril del 2025). *Definicion de la diabetes*, enlace https://www.who.int/es/news-room/fact-sheets/detail/diabetes
        2. National Institute of Diabetes and Digestive and Kidney Disease (NIH). (9 Abril del 2025). *¿Qué es la Diabetes?*, enlace https://www.niddk.nih.gov/health-information/informacion-de-la-salud/diabetes/informacion-general/que-es
        3. Share Care. (9 Abril del 2025). *The Cost of Diabetes in the U.S.: Economic and Well-Being Impact*, enlace https://wellbeingindex-sharecare-com.translate.goog/diabetes-us-economic-well-being-impact/?_x_tr_sl=en&_x_tr_tl=es&_x_tr_hl=es&_x_tr_pto=tc
        4. Centers for Disease Control and Prevention. (CDC). (2025, abril 9). *Behavioral Risk Factor Surveillance System*. Recuperado de [https://www.cdc.gov/brfss/about/brfss_faq.htm](https://www.cdc.gov/brfss/about/brfss_faq.htm)  
        5. Centers for Disease Control and Prevention. (CDC). (2015). *2015 calculated variables: age*. Recuperado de [https://www.cdc.gov/brfss/annual_data/2015/pdf/2015_calculated_variables_version4.pdf](https://www.cdc.gov/brfss/annual_data/2015/pdf/2015_calculated_variables_version4.pdf)  
        6. Wikipedia. (n.d.). (21 de Abril de 20205). *Body mass index*. Recuperado de [https://en.wikipedia.org/wiki/Body_mass_index](https://en.wikipedia.org/wiki/Body_mass_index)  
        """
    )