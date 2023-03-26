import streamlit as st

# Fonctions définies
from presentation_streamlit import presentation
from datasets_streamlit import datasets
from visualisation_streamlit import visualisation
from modelisation_streamlit import modelisation

def main():
    
    # Image CO2
    from PIL import Image
    image = Image.open('Data\CO2_wide.jpg')
    st.image(image)
    
    # Titre
    st.markdown("<h2 style='text-align: center;'>PyCars : Prédictions des émissions de CO2</h2>", unsafe_allow_html=True)
    

    
    # List of pages
    liste_menu = ["Présentation du projet", "Datasets", "Exploration et visualisation", "Machine Learning"]

    # Sidebar
    menu = st.sidebar.selectbox("Menu", liste_menu)

    # Page navigation
    if menu == liste_menu[0]:
        presentation()
    if menu == liste_menu[1]:
        datasets()
    if menu == liste_menu[2]:
        visualisation()
    if menu == liste_menu[3]:
        modelisation()


if __name__ == '__main__':
    main()