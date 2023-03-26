import streamlit as st
import pandas as pd

st.set_page_config(page_title="Data Collection", page_icon="💽")
st.sidebar.header("Data Collection")
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
st.title('Collection des données')
st.subheader("Données brutes")

st.write("Les données brutes téléchargées contiennent un total de **17 092 images de cellules sanguines saines**, qui ont été acquises à l'aide de l'analyseur CellaVision DM96 dans le laboratoire central de l'hôpital clinique de Barcelone.")

st.write("L'ensemble de données est organisé selon les **huit groupes** suivants : **neutrophiles, éosinophiles, basophiles, lymphocytes, monocytes, granulocytes immatures** (promyélocytes, myélocytes et métamyélocytes), **érythroblastes et plaquettes.**")

random_classes = Image.open('streamlit/random_classes.png')
st.image(random_classes)

st.write("Cet premier aperçu met en évidence plusieurs **caractéristiques** qui peuvent **différencier les classes de globules blancs** :")
col1, col2 = st.columns(2)
with col1:
    st.write("- la **taille** des cellules")
    st.write("- la **forme** des cellules")
    st.write("- la **couleur et l'opacité** du cytoplasme")
with col2:
    st.write("- la **taille** du noyau")
    st.write("- la **forme** du noyau")

st.write("")
st.write("La taille des images peut varier légèrement mais la grande majorité est au format **360 x 363 pixels**, en **JPG**, et ont toutes été annotées par des experts en pathologie clinique. Les images ont été capturées chez des personnes ne souffrant pas d'infection, de maladie hématologique ou oncologique et n'ayant reçu aucun traitement pharmacologique au moment de la prise de sang.")

st.write("Les images venant d'une même source, les **luminosités et teintes de celles-ci sont similaires**.")

st.write("Notons que plusieurs éléments peuvent venir **parasiter l'information importante** :")
st.write("- Il y a des **globules rouges en fond**, dont le nombre et l'aspect peuvent différer fortement d'une image à l'autre, et on constate la **possibilité d'avoir plusieurs cellules colorées** sur une même image.")

st.write("A partir de ces images, j'ai choisi de générer **un Dataframe avec deux labels** : le premier label représente les **classes de cellules (8 classes)** et le second représente les **sous-classes (11 sous-classes)**. Le label des classes correspond au nom du dossier dans lequel se trouve l'image et le label des sous-classes correspond à la première partie du nom de chaque image, avant séparation par un '_'.")

with st.expander("Afficher/Cacher le code pour la génération du Dataframe"):
    st.code("""#Création du Dataframe et récupération des chemins d'images et labels

def generate_df_dask(path):
    
    path = Path(path)
    
    df = pd.DataFrame()
    df['img_path'] = [str(image_path) for ext in ['jpg', 'tiff', 'png'] 
                      for image_path in path.glob(f'**/*.{ext}')]
    
    df['classes'] = [image_path.parts[-2] for ext in ['jpg', 'tiff', 'png'] 
                   for image_path in path.glob(f'**/*.{ext}')]

    df['sub_classes'] = [image_path.stem.split('_')[0] 
                     for ext in ['jpg', 'tiff', 'png'] for image_path in path.glob(f'**/*.{ext}')]        
    
    def add_columns(filename):
        image = cv2.imread(filename)
        temp = pd.DataFrame(index=[0])
        return temp
    
    addcol_bag = bag.from_sequence(df.img_path.to_list()).map(add_columns)
    with diagnostics.ProgressBar():
        res = addcol_bag.compute()
        
    res_df = pd.concat(res).reset_index(drop=True)
    df = df.join(res_df)
    return df""")
    
# Import dataset
df = pd.read_csv("data/dataset.csv")
    
# Affichage dataset
st.subheader("Affichage de quelques lignes du dataset")
st.write('Cliquez sur le bouton "Randomize" pour afficher des lignes aléatoires du dataset.')
if st.button('Randomize'):
    st.dataframe(df.sample(6))
