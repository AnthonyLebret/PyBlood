import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly_express as px
import matplotlib.pyplot as plt
import cv2
from dask import bag, diagnostics
import umap

st.set_page_config(page_title="Data Exploration", page_icon="📊")
st.sidebar.header("Data Exploration")
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

st.title('Data Exploration')
    
# Import dataset
df = pd.read_csv("data/dataset.csv")
df_mini = pd.read_csv("data/dataset_mini.csv")
plot_df = pd.read_csv("data/plot_UMAP.csv")
    
### Graphique 1
st.subheader('Répartition des types cellulaires dans le dataset')
fig = px.histogram(df.classes, labels={'value': 'Classe de globule blanc', 'variable':"Quantité d'images", 'color': 'classes'},
                   text_auto='.3s', color = df.classes)
st.plotly_chart(fig)

st.write("On remarque un **déséquilibre dans la répartition des classes de globules blancs**. Alors que les neutrophiles, les éosinophiles et les granulocytes immatures sont sur-représentés, la proportion de lymphocytes, basophiles, monocytes et érythroblastes sont sous-représentés (2 à 3 fois moins que les autres types cellulaires.")
    
### Graphique 2
st.subheader('Projection UMAP')
st.write("Le graphique interactif suivant montre un échantillon de l'ensemble de données après **réduction de dimension à l'aide d'une projection UMAP**.")

fig_3d = px.scatter_3d(data_frame = plot_df, x='X_UMAP', y='Y_UMAP', z='Z_UMAP', color=df_mini.classes, labels={'color': 'classes'})
fig_3d.update_traces(marker_size=5)
fig_3d.update_layout(height=800)
st.plotly_chart(fig_3d)

st.write("On remarque que les **plaquettes (PLATELET) et les érythroblastes sont isolés** du reste de la population cellulaire. On peut donc s'attendre à ce que les modèles aient **plus de facilité à prédire ces classes**.")
