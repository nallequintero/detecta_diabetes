import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Configuración de la página (debe ser lo primero de Streamlit)
st.set_page_config(
    page_title="DetectaDiabetes - Predictor de Riesgo",
    page_icon="🏥",
    layout="wide"
)

# --- Sidebar para navegación ---
page = st.sidebar.radio("Navegación", ["Introducción", "Diabetes y sus tipos", "Acerca de la BRFSS", "Impacto económico en USA en 2015", 
                                       "Diabetes en USA", "Diccionario", "Modelos","Modelo seleccionado",
                                       "Importancia de las variables", "Predicción", "Referencias"])

# Cargar el modelo
model = pickle.load(open('/workspaces/detecta_diabetes/app/models/diabetes_histboost_model.pkl', 'rb'))
class_dic = {0: 'diabetes', 1: 'no tiene diabetes'}

if page == "Introducción":
    st.title("Introducción")
    st.markdown("""
    Este proyecto aborda el tema de la diabetes, una enfermedad crónica que afecta a millones de personas en todo el mundo y tiene un impacto significativo en la calidad de vida y en los sistemas de salud. La detección temprana y el análisis de factores de riesgo son esenciales para desarrollar medidas preventivas eficaces. En este contexto, el presente proyecto se centra en el uso de técnicas de *machine learning* para analizar datos del conjunto **BRFSS** (Behavioral Risk Factor Surveillance System). Su objetivo principal es identificar patrones y construir modelos predictivos que permitan estimar el riesgo de diabetes considerando variables como factores demográficos, conductuales y de salud.
    """)
elif page == "Diabetes y sus tipos":
    st.title("Diabetes y sus tipos")
    st.markdown("""
    ### **Diabetes**
    Enfermedad crónica que afecta la forma en que el cuerpo utiliza la glucosa.
    Puede causar daño a órganos y sistemas si no se controla adecuadamente.

    ### **Tipos de diabetes**

    - Tipo 1: El cuerpo no produce insulina (autoinmune).
    - Tipo 2: El cuerpo no usa correctamente la insulina (más común).
    - Gestacional: Ocurre durante el embarazo.
    - Prediabetes: Glucosa elevada, pero no suficiente para diagnóstico.
    - Otros tipos: Monogénica o inducida por daño al páncreas.
    """)
elif page == "Acerca de la BRFSS":
    st.title("Acerca de la BRFSS (Behavioral Risk Factor Surveillance System)")
    st.markdown("""
    El Sistema de Vigilancia de Factores de Riesgo Conductuales (BRFSS) es una encuesta telefónica sobre salud que los CDC recopilan anualmente. Cada año, la encuesta recopila respuestas de más de 400,000 estadounidenses sobre conductas de riesgo para la salud, enfermedades crónicas y el uso de servicios preventivos. Se lleva a cabo anualmente desde 1984. Para este proyecto, se utilizó un archivo CSV del conjunto de datos disponible en Kaggle para el año 2015.
    """)
elif page == "Impacto económico en USA en 2015":
    st.title("Impacto económico en USA en 2015")
    st.markdown("""
    En el año 2015, el coste sanitario de la diabetes en Estados Unidos ascendió a más de 400.000 millones de euros. Se estima que dicha cifra se incremente considerablemente a lo largo de los próximos años hasta llegar a superar los 600.000 millones de euros en el año 2030.
    """)
elif page == "Diabetes en USA":
    st.title("Diabetes en USA")
    st.markdown("""
    En el año 2015, el número de casos de diabetes registrados en los Estados Unidos ascendió a más de 35 millones. Se prevé que a lo largo de los próximos años dicha cifra se incremente considerablemente, hasta casi llegar a alcanzar los 55 millones de afectados en 2030.
    """)
elif page == "Diccionario":
    st.title("Diccionario de variables utilizadas en el modelo")
    st.markdown("""
| Variable | Descripción |
|----------|-------------|
| Presión arterial alta | 0: Sin presión arterial alta, 1: Con presión arterial alta |
| Colesterol alto | 0: Sin colesterol alto, 1: Con colesterol alto |
| Fumador | 0: Nunca ha fumado, 1: Ha fumado alguna vez |
| Actividad física | 0: No realiza actividad física regular, 1: Sí realiza |
| Consumo de frutas | 0: No consume frutas diariamente, 1: Sí consume |
| Consumo de vegetales | 0: No consume vegetales diariamente, 1: Sí consume |
| Consumo de alcohol excesivo | 0: No es bebedor excesivo, 1: Es bebedor excesivo |
| Sexo | "Femenino" o "Masculino" |
| Edad | "35-44 años", "45-54 años", "55-64 años" |
| Índice de masa corporal | Índice de Masa Corporal (IMC) |
| Salud general | 1: Excelente, 2: Muy buena, 3: Buena, 4: Regular, 5: Mala |
| Ingreso | 1: Menos de 10,000 USD , 2: 10,000-15,000 USD, 3: 15,000-20,000 USD, 4: 20,000-25,000 USD, 5: 25,000-35,000 USD, 6: 35,000-50,000 USD, 7: 50,000-75,000 USD, 8: Más de 75,000 USD |
| No visitó médico por costo | 0: No tuvo problema de costo, 1: No pudo ver al médico por costo |
| Dificultad para caminar | 0: Sin dificultad, 1: Con dificultad para caminar o subir escaleras |
| Días de mala salud mental | Días de mala salud mental en los últimos 30 días (0-30) |
| Días de mala salud física | Días de mala salud física en los últimos 30 días (0-30) |
| Educación | 1: Básico, 2: No se graduó de la preparatoria, 3: Graduado de la preparatoria, 4: Asistió a la universidad o escuela técnica, 5: Graduado de la universidad o escuela técnica |
    """)
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


elif page == "Predicción":
    # Título y descripción
    st.title("🏥 DetectaDiabetes - Predictor de Diabetes")
    st.markdown("""
    Este proyecto utiliza *machine learning* para analizar datos del BRFSS y estimar el riesgo de diabetes considerando factores demográficos, conductuales y de salud. El objetivo es facilitar la detección temprana y apoyar la prevención.
    """)
    # Definir las opciones del modelo
    highbp_options = {"No": "no", "Sí": "yes"}
    highchol_options = {"No": "no", "Sí": "yes"}
    smoker_options = {"No": "no", "Sí": "yes"}
    physactivity_options = {"No": "no", "Sí": "yes"}
    fruits_options = {"No": "no", "Sí": "yes"}
    veggies_options = {"No": "no", "Sí": "yes"}
    hvyalcoholconsump_options = {"No": "no", "Sí": "yes"}
    sex_options = {"Femenino": "female", "Masculino": "male"}
    age_options = {
        "35-44 años": "entre_35_y_44",
        "45-54 años": "entre_45_y_54",
        "55-64 años": "entre_55_y_64"
    }
    genhlth_options = {
        "Excelente": 1,
        "Muy buena": 2,
        "Buena": 3,
        "Regular": 4,
        "Mala": 5
    }
    income_options = {
        "Menos de $10,000 USD/año": 1,
        "Entre $10,000 y $15,000 USD/año": 2,
        "Entre $15,000 y $20,000 USD/año": 3,
        "Entre $20,000 y $25,000 USD/año": 4,
        "Entre $25,000 y $35,000 USD/año": 5,
        "Entre $35,000 y $50,000 USD/año": 6,
        "Entre $50,000 y $75,000 USD/año": 7,
        "Más de $75,000 USD/año": 8
    }
    # Opciones adicionales
    nodocbccost_options = {"No": "no", "Sí": "yes"}
    diffwalk_options = {"No": "no", "Sí": "yes"}
    # Crear columnas para una mejor organización
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Información Personal")
        sex = st.selectbox("⚧ Sexo", list(sex_options.keys()))
        age = st.selectbox("📅 Edad", list(age_options.keys()))
        bmi = st.number_input(
            "⚖️ Índice de Masa Corporal (IMC)", 
            min_value=10.0, 
            max_value=50.0, 
            value=25.0,
            step=1.0,
            help="El IMC se calcula dividiendo el peso (kg) entre la altura al cuadrado (m²)"
        )

    with col2:
        st.subheader("⚕️ Factores de Riesgo")
        highbp = st.selectbox("🫀 ¿Presión arterial alta?", list(highbp_options.keys()))
        highchol = st.selectbox("🔬 ¿Colesterol alto?", list(highchol_options.keys()))
        smoker = st.selectbox("🚬 ¿Fumador/a?", list(smoker_options.keys()))
        nodocbccost = st.selectbox("🩺 ¿No pudo ver al doctor por costo?", list(nodocbccost_options.keys()))
        diffwalk = st.selectbox("🦵 ¿Dificultad para caminar o subir escaleras?", list(diffwalk_options.keys()))

    # Más factores
    st.subheader("🎯 Hábitos y Estilo de Vida")
    col3, col4 = st.columns(2)

    with col3:
        physactivity = st.selectbox("🏃‍♂️ ¿Actividad física regular?", list(physactivity_options.keys()))
        fruits = st.selectbox("🍎 ¿Consume frutas diariamente?", list(fruits_options.keys()))
        veggies = st.selectbox("🥬 ¿Consume vegetales diariamente?", list(veggies_options.keys()))
        menthlth = st.slider("🧠 Días de mala salud mental (últimos 30 días)", min_value=0, max_value=30, value=0)
        physhlth = st.slider("💪 Días de mala salud física (últimos 30 días)", min_value=0, max_value=30, value=0)

    with col4:
        hvyalcoholconsump = st.selectbox("🍺 ¿Consumo excesivo de alcohol?", list(hvyalcoholconsump_options.keys()))
        genhlth = st.selectbox("🏥 Salud General", list(genhlth_options.keys()))
        income = st.selectbox("💵 Nivel de Ingresos Anual", list(income_options.keys()))
        education_options = {
            "1 - Básico": 1,
            "2 - No se graduó de la preparatoria": 2,
            "3 - Graduado de la preparatoria": 3,
            "4 - Asistió a la universidad o escuela técnica": 4,
            "5 - Graduado de la universidad o escuela técnica": 5
        }
        education = st.selectbox("🎓 Nivel educativo", list(education_options.keys()))

    # Botón para realizar la predicción
    if st.button("🔍 Predicción"):
        try:
            input_dict = {
                'highbp': highbp_options[highbp],
                'highchol': highchol_options[highchol],
                'smoker': smoker_options[smoker],
                'physactivity': physactivity_options[physactivity],
                'fruits': fruits_options[fruits],
                'veggies': veggies_options[veggies],
                'hvyalcoholconsump': hvyalcoholconsump_options[hvyalcoholconsump],
                'sex': sex_options[sex],
                'age': age_options[age],
                'bmi': bmi,
                'genhlth': genhlth_options[genhlth],
                'income': income_options[income],
                'nodocbccost': nodocbccost_options[nodocbccost],
                'diffwalk': diffwalk_options[diffwalk],
                'menthlth': menthlth,
                'physhlth': physhlth,
                'education': education_options[education]
            }
            
            input_df = pd.DataFrame([input_dict])
            prediction = int(model.predict(input_df)[0])
            pred_class = class_dic[prediction]
            
            # Resultados en secciones desplegables
            with st.expander("🔍 Ver Resultados del Análisis", expanded=True):
                if pred_class == 'no tiene diabetes':
                    st.success(f"✅ Resultado: {pred_class.title()}")
                else:
                    st.error(f"⚠️ Resultado: {pred_class.title()}")
            
            with st.expander("⚕️ Factores de Riesgo Identificados"):
                risk_factors_list = []
                if highbp == "Sí": risk_factors_list.append("🫀 Presión arterial alta")
                if highchol == "Sí": risk_factors_list.append("🔬 Colesterol alto")
                if bmi > 30: risk_factors_list.append("⚖️ IMC elevado")
                if smoker == "Sí": risk_factors_list.append("🚬 Tabaquismo")
                if physactivity == "No": risk_factors_list.append("🏃‍♂️ Falta de actividad física")
                if hvyalcoholconsump == "Sí": risk_factors_list.append("🍺 Consumo excesivo de alcohol")
                
                if risk_factors_list:
                    for factor in risk_factors_list:
                        st.markdown(f"• {factor}")
                else:
                    st.markdown("✅ No se identificaron factores de riesgo significativos.")

            with st.expander("💡 Recomendaciones"):
                if pred_class == 'no tiene diabetes':
                    st.markdown("""
                    ✅ Recomendaciones para mantener su salud:
                    * 🏃‍♂️ Mantener un estilo de vida saludable
                    * 👨‍⚕️ Continuar con chequeos médicos regulares
                    * 🥗 Mantener una dieta equilibrada
                    * 🏋️‍♂️ Realizar ejercicio regularmente
                    * ⚖️ Mantener un peso saludable
                    """)
                else:
                    st.markdown("""
                    🚨 Recomendaciones importantes:
                    * 👨‍⚕️ Consultar con un médico lo antes posible
                    * 🔬 Realizar exámenes de glucosa
                    * 🥗 Modificar hábitos alimenticios inmediatamente
                    * 🏃‍♂️ Aumentar significativamente la actividad física
                    * 👨‍⚕️ Considerar consulta con un endocrinólogo
                    * 🫀 Monitorear la presión arterial y el colesterol
                    * ❌ Eliminar hábitos perjudiciales
                    """)

            if bmi > 30:
                with st.expander("ℹ️ Información sobre IMC"):
                    st.info("""
                    💡 **Información sobre el IMC:**
                    Un IMC mayor a 30 indica obesidad, lo cual es un factor de riesgo importante para la diabetes.
                    Considere consultar con un nutriólogo para establecer un plan de alimentación adecuado.
                    """)

        except Exception as e:
            st.error(f"❌ Ha ocurrido un error en el procesamiento: {str(e)}")

    # Nota informativa
    st.markdown("---")
    st.markdown("""
    ⚠️ **Nota**: Esta aplicación es solo para fines educativos y no sustituye el diagnóstico médico profesional. 
    Siempre consulte con un profesional de la salud para decisiones médicas importantes.

    📊 **Sobre el IMC:**
    - ⚖️ Bajo peso: < 18.5
    - ✅ Peso normal: 18.5 - 24.9
    - ⚠️ Sobrepeso: 25 - 29.9
    - 🚨 Obesidad: ≥ 30
    """)

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