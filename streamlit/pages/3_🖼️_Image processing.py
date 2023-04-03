import streamlit as st
import pandas as pd

st.set_page_config(page_title="Image processing", page_icon="🖼️")
st.sidebar.header("Image processing")
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
st.title("Traitement de l'image")
st.subheader("Objectif")

st.write("L'objectif de l'image processing dans ce projet sera **d'écarter les éléments qui pourraient venir introduire un biais** dans la reconnaissance des différents types cellulaires par les modèles de **machine learning**. Les images seront donc traitées de sorte à **supprimer les globules rouges en fond et à ne conserver que la cellule colorée centrale**.")

st.write("Toutes les étapes de **filtrages, seuillages, transformations morphologiques et convolutions** opérées sur le dataset ont été réalisées en majeure partie avec la **bibliothèque OpenCV**.")
st.write("")

st.subheader("Étapes du traitement d'image")

schema_seg = Image.open('streamlit/seg_white.png')
st.image(schema_seg)
st.markdown("#### Code")
st.write("Vous trouverez ci-dessous mon code pour la partie traitement de l'image.")

with st.expander("Afficher/Cacher le code pour le traitement de l'image"):
    st.code("""def process_image(img):
    
    img_gray = (img[:,:,0]+img[:,:,2])/(2*img[:,:,1])  # (R+G)/(2*B) -> Pour faire ressortir la cellule qui
                                                                          # a une couleur plus bleutée/violacée
    img_gray2 = img_gray.copy()
    for i in range(img_gray2.shape[0]):
        for j in range(img_gray2.shape[1]):

                if img_gray[i][j]<5:    # Si valeur pixel < 5, prend la valeur 0
                          img_gray2[i,j]=0

    ret, thresh = cv2.threshold(img_gray2.astype(np.uint8),0,255,cv2.THRESH_BINARY_INV)  # Seuillage (Thresholding)

    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 15) # Ouverture (Opening -> erosion + dilatation)
    
    # Zone de background sûre
    sure_bg = cv2.dilate(opening, kernel, iterations=10) # Dilatation

    cell_mask = cv2.bitwise_not(sure_bg) # Inversion des pixels
    
    contour,hier = cv2.findContours(cell_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE) # Recherche des contours de la cellule
    for cnt in contour:
        cv2.drawContours(cell_mask,[cnt],0,255,-1) # Dessin des contours de la cellule

    sure_bg = cv2.bitwise_not(cell_mask) # Inversion des pixels

    img_gray3 = np.ones(sure_bg.shape)*255 # Matrice de pixels blancs    
    ret, thresh2 = cv2.threshold(sure_bg, 127, 255, 0) # Seuillage du background (élimination du bruit)
    
    contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Recherche des contours du background
    image = cv2.drawContours(img_gray3, contours, -1, (0, 255, 0), 1) # Dessin des contours du background

    img_gray4 = np.ones(sure_bg.shape)*255 # Matrice de pixels blancs
    
    # Recherche du contour central
    m=[]
    for ip in range(len(contours)):
        x=[contours[ip][:,:,0].mean(),contours[ip][:,:,1].mean()]
        m.append(np.sqrt((x[0]-sure_bg.shape[0]/2)**2+ (x[0]-sure_bg.shape[0]/2)**2))
        
    if len(m)>1:    # Si len(m) > 1, il s'agit du contour central
        m=m[1:]
        contour_center=contours[np.argmin(m)+1]

        image2=cv2.drawContours(img_gray4, [contour_center], -1, (0, 255, 0), 1) # Dessin du contour central
        dist_transform = cv2.distanceTransform(image2.astype(np.uint8), cv2.DIST_L2,5) # Algorithme de distanceTransform

        ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0) # Masque du foreground (premier plan)
        sure_fg = scipy.ndimage.binary_fill_holes(255-sure_fg).astype(int) # Remplissage des trous éventuels
        img_seg = img.copy()                 
        for k in [0,1,2]:
            for i in range(img_seg.shape[0]):
                for j in range(img_seg.shape[1]):
                    if sure_fg [i,j]==0 :  # Si le pixel est dans le background
                        img_seg[i,j,k]=0  # Il prend la valeur 0
                        
        return 1,img_seg   # segmentation réussie
    
    elif len(m)==1:    # Si len(m) = 1, il ne s'agit pas du contour central
        contour_center=contours[np.argmin(m)]

        image2=cv2.drawContours(img_gray4, [contour_center], -1, (0, 255, 0), 1) # Dessin du contour central
        dist_transform = cv2.distanceTransform(image2.astype(np.uint8), cv2.DIST_L2,5) # Algorithme de distanceTransform

        ret, sure_fg = cv2.threshold(dist_transform,0.01*dist_transform.max(),255,0) # Masque du foreground (premier plan)
        sure_fg = scipy.ndimage.binary_fill_holes(255-sure_fg).astype(int) # Remplissage des trous éventuels
        img_seg = img.copy()              
        for k in [0,1,2]:
            for i in range(img_seg.shape[0]):
                for j in range(img_seg.shape[1]):
                    if sure_fg [i,j]!=0 :  # Si le pixel n'est pas dans le background
                        img_seg[i,j,k]=0  # Il prend la valeur 0                       
        
        return 1,img_seg   # segmentation réussie
    else: 
        return 0,img  # segmentation impossible, retour de l'image d'entrée""")
