import streamlit as st

# Custom CSS to set background image
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("img/bg.svg");
    background-size: cover;
}
</style>
"""

# Apply background image
st.markdown(page_bg_img, unsafe_allow_html=True)

# Rest of your existing code
st.set_page_config(
    page_title="Mis Proyectos",
    page_icon=":bar_chart:",
    layout="wide"
)
st.title("Portfolio Data Analyst & Machine Learning -- David Rojo")
st.write(
    """
    Esta aplicación contiene varios proyectos interesantes. Usa el menú lateral
    para navegar entre las páginas y explorar cada proyecto en detalle.
    """
)