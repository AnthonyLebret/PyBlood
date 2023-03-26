# import des packages nécessaires à la modélisation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly_express as px

st.set_page_config(page_title="Classification", page_icon="🗂️")
st.sidebar.header("Classification")
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
image = Image.open('data/PyBlood.jpg')
st.image(image)

# Titre
st.title("Machine Learning")

# Classification
st.header("Classification")
st.subheader("Réduction de dimension")
st.write("Afin de pouvoir entraîner **plusieurs modèles de machine learning** sur l'ensemble des données (~17 000 images) en un temps raisonnable, j'ai opté pour une **réduction de dimension via UMAP**, qui a le double avantage de pouvoir être employé pour des **visualisations** mais aussi pour faire des **réductions non linéaires générales**. La taille des images a également été réduite aux dimensions 256x256.")

# Import dataset
df = pd.read_csv("data/dataset.csv")
plot_df = pd.read_csv("data/plot_UMAP_2d.csv")

fig_2d = px.scatter(data_frame = plot_df, x='X_UMAP', y='Y_UMAP', color=df.classes, labels={'color': 'classes'})
fig_2d.update_traces(marker_size=2)
st.plotly_chart(fig_2d)

st.subheader("Optimisation des hyperparamètres")
st.write("Toujours dans une démarche de **diminution des temps d'exécution**, les hyperparamètres ont été optimisés par une **méthode hybride de recherche en grille (GridSearchCV) et recherche bayésienne**. Cette méthode est généralement **aussi performante** qu'une optimisation par recherche en grille stricte mais est **bien plus rapide**.")
with st.expander("Afficher/Cacher le code pour l'optimisation des hyperparamètres"):
    st.code("""# Optimisation des paramètres

def best_model_Ray_bayes(X_train, y_train, name, model):    
    '''run gridsearchCV pipeline with bayesian method on models
    Args:
        -model: initiated model 
        -name : name of model as str
    return list of best estimator and table of results
    '''

    best_model_stack = list()
    results_cv = dict()
    
    def grid_csv(params, early_stopping=False):

        GSCV = TuneSearchCV(model, params, 
                            search_optimization='bayesian',
                            n_trials=7,
                            early_stopping=early_stopping,
                            max_iters=10,
                            scoring = 'accuracy',
                            loggers=['csv'],
                            cv = 5, n_jobs=7, verbose=1)

        best_clf = GSCV.fit(X_train, y_train)

        best_hyperparams = best_clf.best_params_
        best_score = best_clf.best_score_
        estimator = best_clf.best_estimator_
        print(f'Mean cross-validated accuracy score of the best_estimator:{best_score:.3f}')
        print(f'with {best_hyperparams} for {estimator}')
        table = best_clf.cv_results_
        results_cv[name] = table

        return estimator
        

    if name == 'LR':
        params = {'C' : tuple([0.001, 0.01, 0.1, 1.]),
                 'penalty' : tuple(['l1', 'l2'])} 
        best_model_stack.append(grid_csv(params))
              
    if name == 'KNN':
        params = {'n_neighbors' : tuple(np.arange(5, 100, 5)),
                 'weights' : tuple(['uniform', 'distance']),
                 'algorithm' : tuple(['ball_tree', 'kd_tree', 'brute', 'auto'])} 
        best_model_stack.append(grid_csv(params))
    
    if name == 'SVM':
        params = {'kernel' : tuple(['linear', 'poly', 'rbf', 'sigmoid']),
                 'C' : tuple(np.arange(0.01, 1, 0.02))} 
        best_model_stack.append(grid_csv(params))


    if name == 'RF': 
        params = {'n_estimators' : tuple(np.arange(5, 200, 10)),
                  'max_features' : tuple(['auto', 'sqrt', 'log2']),
                  'max_depth' : tuple(np.arange(3, 15, 1)),
                  'min_weight_fraction_leaf': tuple(np.arange(0, 0.6, 0.2))
                 } 
        best_model_stack.append(grid_csv(params))

        
    return best_model_stack, results_cv""")

st.subheader("Entraînement des modèles")
st.write("Les données **réduites après UMAP** ont été utilisées afin d'entraîner **4 modèles de classification** :")
col1, col2 = st.columns(2)
with col1:
    st.write("- **Régression logistique (LR)**")
    st.write("- **Algorithme des K plus proches voisins (KNN)**")
with col2:
    st.write("- **Machine à vecteurs de support (SVM)**")
    st.write("- **Algorithme des forêts aléatoires (RF)**")

st.subheader("Résultats")
st.markdown("#### Sur les images non-segmentées")

st.write("L'exécution de la pipeline d'entraînement des modèles pouvant prendre **plus d'une heure**, celle-ci ne sera **pas exécutée sur cette page**. Néanmoins, il est intéressant de noter que **les scores varient de plusieurs pourcentages d'une exécution à une autre**. Cette **variabilité** ne semble pas être induite par l'optimisation des hyperparamètres (ceux-ci sont relativement constants) mais vraisemblablement par **les images sélectionnées comme échantillons d'entraînement et de test**")

st.write("**Les meilleurs scores** sont obtenus par le modèle de **forêts aléatoires**. L'accuracy et le F1-Score sont de **0.75**. Comme attendu après visualisation de la projection UMAP, **les plaquettes sont correctement classées à 95%** (F1-Score). Néanmoins, les classes basophiles et granulocytes immatures (IG) viennent tirer le score moyen vers le bas : ils sont correctement prédits à hauteur de **59%** pour le premier et **65%** pour le second.")

rf = Image.open('streamlit/rf.png')
st.image(rf)

with st.expander("Afficher/Cacher la matrice de confusion et le rapport de classification du modèle de régression logistique"):
    lr = Image.open('streamlit/lr.png')
    st.image(lr)
with st.expander("Afficher/Cacher la matrice de confusion et le rapport de classification du modèle des K plus proches voisins"):
    knn = Image.open('streamlit/knn.png')
    st.image(knn)
with st.expander("Afficher/Cacher la matrice de confusion et le rapport de classification du modèle SVM"):
    svm = Image.open('streamlit/svm.png')
    st.image(svm)

st.markdown("#### Sur les images segmentées")

st.write("A la différence des scores sur les images brutes, **les meilleurs scores obtenus sur les images segmentées** sont détenus par le modèle des **K plus proches voisins**. L'accuracy et le F1-Score sont de **0.57-0.58**. De façon globale, les résultats sont beaucoup moins bon sur les images après segmentation.")

st.write("**On peut toutefois noter que les plaquettes sont correctement classées à 96%**, elles sont donc aussi bien classées (même un peu mieux, sûrement dû à la variabilité du jeu de test) que pour les modèles entraînés sur des images non-segmentées.")

knn_seg = Image.open('streamlit/knn_seg.png')
st.image(knn_seg)

with st.expander("Afficher/Cacher la matrice de confusion et le rapport de classification du modèle de régression logistique"):
    lr_seg = Image.open('streamlit/lr_seg.png')
    st.image(lr_seg)
with st.expander("Afficher/Cacher la matrice de confusion et le rapport de classification du modèle SVM"):
    svm_seg = Image.open('streamlit/svm_seg.png')
    st.image(svm_seg)
with st.expander("Afficher/Cacher la matrice de confusion et le rapport de classification du modèle Random Forest"):
    rf_seg = Image.open('streamlit/rf_seg.png')
    st.image(rf_seg)
