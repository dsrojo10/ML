import streamlit as st

st.title("Proyecto 1: Recomendador de Canciones")
st.write("Aquí puedes explorar un recomendador de canciones utilizando Supabase y Elixir.")

# Ejemplo de contenido
st.subheader("Descripción")
st.write("""
Este proyecto utiliza Spotify Embed para mostrar artistas basados en géneros musicales.
""")

# Agrega contenido interactivo, como ejemplos o visualizaciones.
st.subheader("Prueba una recomendación")
genero = st.selectbox("Selecciona un género", ["Pop", "Rock", "Jazz"])
st.write(f"Has seleccionado el género: {genero}")
