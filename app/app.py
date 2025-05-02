import streamlit as st
import pickle
import pandas as pd

# Cargar el modelo
model = pickle.load(open('/workspaces/detecta_diabetes/app/models/diabetes_histboost_model.pkl', 'rb'))
class_dic = {0: 'diabetes', 1: 'no tiene diabetes'}

st.set_page_config(page_title="DetectaDiabetes", layout="centered")
st.markdown('<style>description{color:white;}</style>', unsafe_allow_html=True)
st.markdown(
    "<description>Esta aplicación predice el riesgo de diabetes basado en indicadores de salud y estilo de vida del BRFSS 2015. "
    "Por favor, ingresa la información solicitada para obtener una predicción.</description>",
    unsafe_allow_html=True
)

# --- Sidebar para navegación ---
page = st.sidebar.radio("Navegación", ["Predicción", "Información", "Visualización"])

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

elif page == "Información":
    st.title("Información del Proyecto")
    st.markdown("Aquí puedes poner información general, instrucciones, etc.")
    #st.image("ruta/a/tu/imagen1.png", caption="Ejemplo de imagen")

elif page == "Visualización":
    st.title("Visualización de Resultados")
    st.markdown("Aquí puedes mostrar gráficos o imágenes de resultados.")
    #st.image("ruta/a/tu/imagen2.png", caption="Resultados del modelo")