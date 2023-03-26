import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly_express as px
import time

st.set_page_config(page_title="Pourquoi embaucher Anthony", page_icon="👨‍🎓")
st.sidebar.header("Analyse très objective du candidat")
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
st.title('Analyse très objective du candidat')

st.subheader("Les raisons de recruter Anthony")

col1, col2 = st.columns(2)
with col1:
    st.success("- Il est hyper motivé et déterminé")
    st.success("- Il est sérieux mais ne se prend pas au sérieux")
    st.success("- Il a un fort esprit d'équipe")
    st.success("- Il est passionné par les intelligences artificielles ET la biologie")
with col2:
    st.success("- Il amène souvent le petit-déjeuner")
    st.success("- Il est toujours de bonne humeur")
    st.success("- C'est un bosseur")
    st.success("- Il ne parle pas toujours de lui à la 3ème personne")
    
if st.checkbox("Afficher plus de raisons de recruter Anthony"):
    col1, col2 = st.columns(2)
    with col1:
        st.success("- Il est très curieux")
        st.success("- Il baisse rarement les bras (ce qui n'est pas très pratique)")
        st.success("- Il s'est challengé à réaliser ce projet en 5 jours et a ainsi respecter la deadline qu'il s'était lui-même fixé")
    with col2:
        st.success("- Il a de l'auto-dérision")
        st.success("- Il ne fait pas de bruit quand il mange")
        st.success("- Il a du recul sur ses défauts : impatient, maladroit, obstiné (là il n'y a plus de doutes), discret...")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("On est convaincu, et on aimerait te voir pour un nouvel entretien"):
            st.balloons()
            st.success("Une notification a été envoyé à M.Anthony Lebret")
            time.sleep(2.5)
            st.success("Non je plaisante, je ne sais pas encore comment coder cette fonction. Vous pouvez m'envoyer un mail ou me contacter directement sur mon téléphone.")
    with col2:
        if st.button("L'équipe Diamidex a décidé de t'éliminer, et leur sentence est irrévocable"):
            st.success("Merci d'avoir pris le temps de consulter ce mini-projet. J'ai travaillé d'arrache-pied directement après avoir reçu le mail de refus en espérant que cela puisse vous convaincre de ma motivation et mettre en valeur mes compétences. Je reste à votre disposition si vous avez la moindre question.   \n Excellente continuation, Anthony.")
        
                
st.subheader("Les raisons de ne pas recruter Anthony")
if st.button("Afficher les raisons de ne pas recruter Anthony"):
    with st.spinner('Wait for it...'):
        time.sleep(2)
        st.exception("ArgumentsNotFoundError - No reasons found")
