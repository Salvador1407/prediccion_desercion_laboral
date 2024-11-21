import pandas as pd
import joblib
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Deserción Laboral",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Cargar el modelo y el dataset
model_path = 'modelo_regresion_XGBOOST.pkl'
data_path = 'df_data_fs.pkl'

pipeline = joblib.load(model_path)
df = pd.read_pickle(data_path)

# Obtener los nombres de las columnas predictoras
target_column = 'Deserción'  # Ajustar si tiene otro nombre
predictors = df.drop(columns=[target_column])
column_names = predictors.columns.tolist()

# Título de la aplicación
st.title("Predicción de Deserción Laboral")
st.write(
    "Ingrese los valores de las variables para predecir la deserción laboral de un empleado."
)

# Crear entradas dinámicamente para las variables
input_data = {}
for col in column_names:
    if col == "HorasExtra":  # Para la columna categórica
        input_data[col] = st.selectbox(f"{col} (Yes/No)", options=["Yes", "No"])
    else:
        input_data[col] = st.number_input(f"{col}", value=0)

# Botón para realizar la predicción
if st.button("Predecir"):
    try:
        # Convertir las entradas en un DataFrame
        input_df = pd.DataFrame([input_data], columns=column_names)
        
        # Asegurarse de que los valores categóricos estén codificados correctamente
        input_df["HorasExtra"] = input_df["HorasExtra"].map({"Yes": 1, "No": 0})
        
        # Verificar si hay valores faltantes
        if input_df.isnull().values.any():
            raise ValueError("Algunas columnas tienen valores no válidos o están vacías.")
        
        # Asegurarse de que los tipos coincidan
        for col in input_df.columns:
            expected_type = df[col].dtype
            input_df[col] = input_df[col].astype(expected_type)
        
        # Realizar la predicción
        prediction = pipeline.predict(input_df)[0]
        result = "Sí" if prediction == 1 else "No"
        
        # Mostrar el resultado con colores
        if result == "Sí":
            st.markdown(
                f"<h4 style='color: red;'>¿Deserción? {result}  :(</h4>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<h4 style='color: green;'>¿Deserción? {result} :)</h4>",
                unsafe_allow_html=True,
            )
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
