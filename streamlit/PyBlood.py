import streamlit as st

st.set_page_config(
    page_title="PyBlood",
    page_icon="🩸"
)
st.sidebar.header("PyBlood")
st.sidebar.info("Auteur : \n\n - Anthony LEBRET [Linkedin](https://linkedin.com/in/anthony-lebret-a7aabb176)")
st.sidebar.info("Données : [Mendeley Data - A dataset for microscopic peripheral blood cell images for development of automatic recognition systems](https://data.mendeley.com/datasets/snkd93bnjr/1)")

# Cacher "Made with Streamlit"
hide_footer_style = """ 
    <style>
    footer {visibility: hidden; }
    </style>
    """
st.markdown(hide_footer_style, unsafe_allow_html=True)

# Image PyBlood
from PIL import Image
image = Image.open('streamlit/PyBlood.jpg')
st.image(image)
    
# Titre
st.markdown("<h1 style='text-align: center;'>PyBlood : Classification des cellules sanguines</h1>", unsafe_allow_html=True)

# Project PyBlood
st.write("Projet personnel réalisé en l'espace de 5 jours dans le but de mettre en pratique mes **compétences en data science** et de **convaincre Arnold** de ma capacité à être rapidement opérationnel sur la partie **traitement d'image**.")
st.write("Auteur :")
st.write("- Anthony LEBRET", "[Linkedin](https://linkedin.com/in/anthony-lebret-a7aabb176)")
st.write("Source de données :", "https://data.mendeley.com/datasets/snkd93bnjr/1")
st.write()

# Contexte
st.header("Contexte et problématique")
st.write("**L'identification** et la **classification** des leucocytes, des plaquettes et des érythrocytes sont **cruciales** pour le diagnostic de plusieurs **maladies hématologiques, telles que les maladies infectieuses ou la leucémie**.")
st.write("L'évaluation visuelle et qualitative des frottis sanguins est souvent nécessaire au diagnostic. Cependant, **l'identification manuelle des cellules du sang est difficile, longue, sujette aux erreurs et nécessite la présence d'un spécialiste qualifié**.")
         
st.header("Objectif")
st.write("Etablir un **système de reconnaissance** des globules blancs reposant sur la **segmentation d'image** et la **classification** par des modèles d'apprentissage automatique. (Et par la même occasion, tenter d'obtenir un nouvel entretien chez Diamidex 🙂.)")

# Étapes du projet
st.header("Étapes du projet")
schema = Image.open('streamlit/Schéma PyBlood.png')
st.image(schema)

st.info("Pour accéder aux analyses et graphiques interactifs (Data Collection, Data Exploration, Image Segmentation, Machine learning : Classification et Conclusion), cliquez dans le menu de gauche.")
