import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression

df_music= pd.read_csv('df_music.csv.zip', sep=',')



st.markdown("<h1 style='text-align: center; color : black;'>BcomeBetter</h1>", unsafe_allow_html=True)
st.text("\n")
st.text("\n")
#st.image('gorille_music.jpg', width=700)

st.image('Pictures/gorille_music.jpg', width=700)
         
for i in range(12):
    st.text("\n")  

# Intro au projet :
st.markdown("<h5 style='text-align: left; color : black;'>Histoire du projet</h5>", unsafe_allow_html=True)
st.write(
"""
 Pour un musicien professionnel, créer et concevoir une chanson, demande beaucoup d'investissements : 
 - du temps 
 - de l'argent
 - du matériel
""")
st.write('Avec toujours l\'incertitude : "Est-ce que ma chanson aura du succès ?"')
st.markdown("<h5 style='text-align: left; color : black;'>Notre solution</h5>", unsafe_allow_html=True)
st.write(
"""
 Accompagner les musiciens professionnels à trouver leur public 
""")

st.markdown("<h5 style='text-align: left; color : black;'>Comment</h5>", unsafe_allow_html=True)
st.write(
"""
 Nous prédisons la popularité de votre chanson grâce à l'analyse des caractéristiques techniques de votre chanson.   
""")
st.write('Etape 1 :')
st.write('En tant que musicien, vous sélectionnez le genre de votre composition.')
st.text("\n")
st.write('Etape 2 :')
st.write("Vous entrez les caractéristiques techniques de votre chanson. Ce sont par exemple, la danceability (capacité à pouvoir danser sur la chanson), le tempo, l'acousticness(indique si la piste est acoustique)...")
st.text("\n")
st.write('Etape 3 :')
st.write('Notre algorithme analyse les paramètres de votre chanson et les compare à la moyenne des paramètres de votre genre musicale. Cela vous donne visuellement votre positionnement technique par rapport aux autres chansons du même genre.')
st.text("\n")
st.write('Etape 4 :')
st.write('Dans un second, avec ces mêmes informations, notre algorithme prédit la popularité de votre titre en se basant sur la popularité des autres chansons de votre genre musical. Derrière cette prédiction nous avons mis en place un algorithme de machine learning en python.')
st.write('Pour contextualiser votre popularité prédite, nous vous donnons la popularité seuil à atteindre en fonction de votre genre musical. Celle-ci correspond au seuil en-dessous duquel se trouvent 75% des chansons de ce genre (3ème quartile de la variable "popularity" dans le dataset).')
st.text("\n")
st.write('Etape 5 :')
st.write('Enfin nous vous permettons de modifier certains critères techniques de votre chanson pour atteindre la popularité seuil définie au-dessus. Pour cela, il vous suffit de bouger les curseurs de 4 caractéristiques de votre piste.')

st.text("\n")
st.markdown("<h3 style='text-align: center; color : black;'>Suivez-nous, nous allons vous aider à passer votre chanson de l'anonymat à la célébrité</h3>", unsafe_allow_html=True)

st.text("\n")

col1, col2, col3 = st.columns([1,6,1])
with col1:
    st.write("")
with col2:
    st.image("Pictures/etoiles filantes.jpg", width=500)
with col3:
    st.write("")

st.text("\n")



# Normalize data between 0 and 1 :
df_music.loc[(df_music.genre == "Children's Music")|(df_music.genre == "Children’s Music"),'genre'] = "Children"
df_music['genre_lower'] = df_music['genre'].apply(lambda x : x.lower())
df_music['pop_scaled'] = df_music['popularity'].apply(lambda x : x/100)
df_music['tempo_scaled'] = df_music['tempo']/df_music['tempo'].max() 
df_music['time_sig_number'] = df_music['time_signature'].apply(lambda x : int(x[0])*0.2)
df_music['mode_number'] = df_music['mode'].apply(lambda x : 1 if x == 'Major' else 0)
df_music['loudness_scaled'] = abs((df_music['loudness']-df_music['loudness'].min())/(df_music['loudness'].max()-df_music['loudness'].min()))
df_music_NN = df_music[['artist_name',	'track_name',	'track_id',	'genre_lower', 'popularity',	'acousticness',	'danceability',	'energy',	'instrumentalness',	
                        'liveness',	'speechiness',	'valence',	'time_sig_number',	'mode_number', 'tempo_scaled',	'loudness_scaled']]


Flag = True
# Select your song's genre :
st.markdown("<h3 style='text-align: left; color : black;'>Place à la musique !</h3>", unsafe_allow_html=True)
st.text("\n")
st.write("Afin de simuler l'utilisation de notre outil par un musicien, nous avons chargé 3 genres musicaux ainsi que leurs chansons respectives.")

choose_genre = st.selectbox('Choisissez le genre de votre musique:', [' ','Rock','Pop','R&B'])

st.text("\n")
# Select a song from the list :
if choose_genre != ' ':
    choose_song = st.selectbox("Choisissez la chanson correspondant au genre sélectionné, pour que nous puissions évaluer votre future popularité (avec ce choix nous simulons le dépôt d'une chanson par un musicien)", ['Choisissez votre chanson','Chanson Pop','Chanson R&B','Chanson Rock'])
    song = [np.nan, np.nan, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0 ]

    if choose_song != "Choisissez votre chanson":
        
        if choose_song == 'Chanson Pop': 
            song = [choose_genre, np.nan, 0.0421, 0.726,0.654,0.000000, 0.1060, 0.0917, 0.335, 0.8, 0, 0.699864, 0.839256]
        
        st.text("\n")
        st.write("Comme expliqué précédemment à l'étape 3, voici graphiquement le positionnement technique de votre chanson par rapport aux autres du même genre.")
        
        #user's input
        if choose_genre == 'Pop':
            st.text("\n")
            st.image("Pictures/image graph Pop et chanson 1 pop.png", width=780)          
            st.text("\n")
            st.text("\n")

            with st.expander('Lancez votre prédiction'):
                st.write("Dans le tableau ci-dessous, vous trouverez le rappel du genre sélectionné (genre), la note de popularité (popularity) que nous vous prédisons ainsi que la popularité seuil à atteindre en fonction de votre genre musical (popularity threshold). Atteindre ce seuil vous place dans les 25% de chansons les plus populaires du genre.")           
                st.text("\n")
        #creating df based on user's input
                df_music_NN_genre = df_music_NN[df_music_NN['genre_lower']==choose_genre.lower()]

                #preparing nearest neighbors
                columns = ['acousticness',	'danceability',	'energy',	'instrumentalness',	'liveness',	'speechiness',	'valence',
                    'mode_number',	'time_sig_number', 'tempo_scaled',	'loudness_scaled']
                y_NN = df_music_NN_genre['popularity']
                X_NN = df_music_NN_genre[columns]

                #fitting model
                model_NN = NearestNeighbors(n_neighbors = 3)
                model_NN.fit(X_NN)

                #user's song's data => greather than 3rd quartile
                data_NN_pop = pd.DataFrame(data = [song], 
                                        columns  = ['genre_lower',	'popularity',	'acousticness',	'danceability',	'energy',	'instrumentalness',	'liveness',	'speechiness',	'valence',	'time_sig_number',	'mode_number', 'tempo_scaled',	'loudness_scaled'], index = [250000] )

                #getting neighbors
                distances, indices = model_NN.kneighbors(data_NN_pop[columns])

                #getting results in dataframe with the 3 nearest neighbors
                output_NN = df_music_NN_genre.iloc[[indices[0][0], indices[0][1], indices[0][2]], :]

                #geeting the distances weighted popularity
                total_dist = (1/distances[0][0] + 1/distances[0][1] + 1/distances[0][2]) 
                output_popularity = (output_NN['popularity'].iloc[0]/(distances[0][0] * total_dist)) + (output_NN['popularity'].iloc[1]/(distances[0][1] *total_dist))+ (output_NN['popularity'].iloc[2]/(distances[0][2] * total_dist))


                #getting all results in a beautiful dataframe
                pop_and_confidence = pd.DataFrame(data = [choose_genre, distances[0][0], round(output_popularity)])
                pop_and_confidence = pop_and_confidence.T
                pop_and_confidence.rename(columns = {0 : 'Genre', 1: 'Distance', 2 : 'Popularity'}, inplace = True)


                #getting confidence based on distance of first nearest neighbors
                #the function to determien which confidence we have
                def confidence(x) : 
                    distance_IC = [0.15, 0.30, 0.60, 0.9]
                    IC = ['Very good', 'Good', 'Be careful', "Something's wrong"]
                    for i in range(len(distance_IC)):
                        if x < distance_IC[i] :
                            x = IC[i]
                            return x
                    return "Don't worry We will help you"

                pop_and_confidence['Confidence'] = pop_and_confidence['Distance'].apply(confidence)
                pop_and_confidence['Popularity Threshold'] = df_music_NN[df_music_NN['genre_lower'] == choose_genre.lower()]['popularity'].quantile(0.75).astype(int)
                
                pop_and_confidence[['Genre','Popularity','Confidence', 'Popularity Threshold']]

                st.text("\n")
                st.markdown("<h2 style='text-align: center; color : black;'>La Célébrité est à vous !</h2>", unsafe_allow_html=True) 
                st.text("\n")

                col1, col2, col3 = st.columns([1,6,1])
                with col1:
                    st.write("")
                with col2:
                    st.image("Pictures/wow.jpg", width=500)
                with col3:
                    st.write("")
                
                st.text("\n")

    if choose_song == 'Chanson R&B': 
        song = [choose_genre, np.nan, 0.0521,	0.726	,0.654,	0.000000,	0.1060,	0.0917,	0.335	,0.8,	0,	0.699864,	0.839256]
                    
        #user's input
        if choose_genre == 'R&B':
            st.text("\n")
            st.image("Pictures/image graph R&B et chanson 2.png", width=780)          
            st.text("\n")
            st.text("\n")

            with st.expander('Lancez votre prédiction'):
                st.text("\n")
                st.write("Dans le tableau ci-dessous, vous trouverez le rappel du genre sélectionné (genre), la note de popularité (popularity) que nous vous prédisons ainsi que la popularité seuil à atteindre en fonction de votre genre musical (popularity threshold). Atteindre ce seuil vous place dans les 25% de chansons les plus populaires du genre.")           
                st.text("\n")
                #TEsting others feature values to go below median
                #creating df based on user's input
                df_music_NN_genre_2 = df_music_NN[df_music_NN['genre_lower']==choose_genre.lower()]

                #preparing nearest neighbors
                columns = ['acousticness',	'danceability',	'energy',	'instrumentalness',	'liveness',	'speechiness',	'valence','time_sig_number', 'mode_number', 'tempo_scaled',	'loudness_scaled']
                y_NN_2 = df_music_NN_genre_2['popularity']
                X_NN_2 = df_music_NN_genre_2[columns]

                #fitting model
                model_NN_2 = NearestNeighbors(n_neighbors = 3)
                model_NN_2.fit(X_NN_2)

                #user's song's data => below median 
                data_NN_pop_2 = pd.DataFrame(data = [song], 
                                            columns  = ['genre_lower',	'popularity',	'acousticness',	'danceability',	'energy',	'instrumentalness',	'liveness',	'speechiness',	'valence',	'time_sig_number',	'mode_number', 'tempo_scaled',	'loudness_scaled'], index = [250000] )

                #getting neighbors
                distances_2, indices_2 = model_NN_2.kneighbors(data_NN_pop_2[columns])

                #getting results in dataframe with the 3 nearest neighbors
                output_NN_2 = df_music_NN_genre_2.iloc[[indices_2[0][0], indices_2[0][1], indices_2[0][2]], :]

                #geeting the distances weighted popularity
                total_dist_2 = (1/distances_2[0][0] + 1/distances_2[0][1] + 1/distances_2[0][2]) 
                output_popularity_2 = (output_NN_2['popularity'].iloc[0]/(distances_2[0][0] * total_dist_2)) + (output_NN_2['popularity'].iloc[1]/(distances_2[0][1] *total_dist_2))+ (output_NN_2['popularity'].iloc[2]/(distances_2[0][2] * total_dist_2))


                #getting all results in a beautiful dataframe
                pop_and_confidence_2 = pd.DataFrame(data = [choose_genre, distances_2[0][0], round(output_popularity_2)])
                pop_and_confidence_2 = pop_and_confidence_2.T
                pop_and_confidence_2.rename(columns = {0 : 'Genre', 1: 'Distance', 2 : 'Popularity'}, inplace = True)

                #getting confidence based on distance of first nearest neighbors
                #the function to determien which confidence we have
                def confidence(x) : 
                    distance_IC = [0.15, 0.30, 0.60, 0.9]
                    IC = ['Very good', 'Good', 'Be careful', "Something's wrong"]
                    for i in range(len(distance_IC)):
                        if x < distance_IC[i] :
                            x = IC[i]
                            return x
                    return "Don't worry We will help you"

                pop_and_confidence_2['Confidence'] = pop_and_confidence_2['Distance'].apply(confidence)
                pop_and_confidence_2['Popularity Threshold'] = df_music_NN[df_music_NN['genre_lower'] == choose_genre.lower()]['popularity'].quantile(0.75).astype(int)
                
                pop_and_confidence_2[['Genre','Popularity','Confidence', 'Popularity Threshold']]

                st.markdown("<h4 style='text-align: left; color : black;'>Le résultat n'est pas au RDV... </h4>", unsafe_allow_html=True) 
                st.markdown("<h4 style='text-align: left; color : black;'>Vous pourriez penser à changer le genre de la musique </h4>", unsafe_allow_html=True)

                #FEATURES TUNING TEST 

                #Creating a dataframe to store every 3 nearest neighbors for each category
                concat_output = pd.DataFrame()

                #creating a dictionnary to store every popularity score for each category
                popularity_scores = {}

                #creating a dictionnary to store every first nearest neighbors  for each category
                all_distances = {}

                targeted_pop = {}

                #user's input
                for genre in df_music['genre_lower'].unique() :

                    #creating df based on user's input
                    df_music_NN_genre_4 = df_music_NN[df_music_NN['genre_lower']==genre]
                    
                    #preparing nearest neighbors
                    columns = ['acousticness',	'danceability',	'energy',	'instrumentalness',	'liveness',	'speechiness',	'valence','time_sig_number', 'mode_number','tempo_scaled',	'loudness_scaled']
                    y_NN_4 = df_music_NN_genre_4['popularity']
                    X_NN_4 = df_music_NN_genre_4[columns]

                    #fitting model
                    model_NN_4 = NearestNeighbors(n_neighbors = 3)
                    model_NN_4.fit(X_NN_4)

                    #user's song's data
                    acousticness = 0.0521
                    danceability = 0.726
                    energy = 0.654
                    instrumentalness = 0.000000
                    liveness = 0.0060
                    speechiness = 0.0917
                    valence = 0.335
                    time_sig_number = 0.8
                    mode_number = 0
                    tempo_scaled = 0.699864
                    loudness_scaled = 0.839256
                    data_NN_pop_4 = pd.DataFrame(data = [[genre, np.nan, acousticness,	danceability, energy,	instrumentalness,	liveness,	speechiness,	valence,	time_sig_number,	mode_number, tempo_scaled,	loudness_scaled]], 
                                                columns  = ['genre_lower',	'popularity',	'acousticness',	'danceability',	'energy',	'instrumentalness',	'liveness',	'speechiness',	'valence',	'time_sig_number',	'mode_number', 'tempo_scaled',	'loudness_scaled'], index = [250000] )
                    
                    #getting neighbors
                    distances_4, indices_4 = model_NN_4.kneighbors(data_NN_pop_4[columns])
                    

                    #geting results
                    output_NN_4 = df_music_NN_genre_4.iloc[[indices_4[0][0], indices_4[0][1], indices_4[0][2]], :]
                    concat_output = pd.concat([concat_output, output_NN_4])

                    #getting popularity
                    total_dist_4 = (1/distances_4[0][0] + 1/distances_4[0][1] + 1/distances_4[0][2]) 
                    output_popularity_4 = ((output_NN_4['popularity'].iloc[0]/(distances_4[0][0] * total_dist_4)) + (output_NN_4['popularity'].iloc[1]/(distances_4[0][1] *total_dist_4))+ (output_NN_4['popularity'].iloc[2]/(distances_4[0][2] * total_dist_4)))
                    
                    #appending popularity to correponding dictionnary
                    popularity_scores[genre] = output_popularity_4 

                    #appending first nearest neighbor's distance to correponding dictionnary
                    all_distances[genre] = distances_4[0][0]
                    targeted_pop[genre] = df_music_NN[df_music_NN['genre_lower'] == genre]['popularity'].quantile(0.75).astype(int)

                #Getting a beautiful dataframe with popularity and first nearest neighbors of user's song for each genre
                #Creating the distance df and renaming columsn
                df_distance = pd.DataFrame(data = [np.array(list(all_distances.keys())),np.array(list(all_distances.values()))]).T
                df_distance.rename(columns = {0 : 'Genre', 1 : 'Distance'}, inplace = True)

                #Creating the popularity df and renaming columsn
                df_popularity = pd.DataFrame(data = [np.array(list(popularity_scores.keys())),np.array(list(popularity_scores.values()))]).T
                df_popularity.rename(columns = {0 : 'Genre_1', 1 : 'Popularity'}, inplace = True)

                df_targeted_popularity = pd.DataFrame(data = [np.array(list(targeted_pop.keys())),np.array(list(targeted_pop.values()))]).T
                df_targeted_popularity.rename(columns = {0 : 'Genre_1', 1 : 'Popularity Threshold'}, inplace = True)

                #Concatenating the last dfs
                df_change_genre = pd.concat([df_distance, df_popularity, df_targeted_popularity], axis = 1)
                df_change_genre = df_change_genre[['Genre', 'Distance', 'Popularity', 'Popularity Threshold']]


                #Adding the confidence columns
                def confidence(x) : 
                    distance_IC = [0.15, 0.30, 0.60, 0.9]
                    IC = ['Very good', 'Good', 'Be careful', "Something's wrong"]
                    for i in range(len(distance_IC)):
                        if x < distance_IC[i] :
                            x = IC[i]
                            return x
                    return ''

                df_change_genre['Confidence'] = df_change_genre['Distance'].apply(confidence)

                #displayig the final results by sorting value 
                reco = df_change_genre[['Genre', 'Popularity', 'Popularity Threshold', 'Confidence']].sort_values(by = 'Popularity', ascending = False, ignore_index=True).head(2)
                st.write(f'Si vous choisissez plutôt le genre',{reco['Genre'][0]},'la popularité de votre chanson serait de',np.around(reco['Popularity'][0]))
                st.write(f'Soit de',np.around(reco['Popularity'][0])-(reco['Popularity Threshold'][0]), 'points supérieur au seuil de popularité requis, avec un indice de confiance de',{reco['Confidence'][0]})
                #st.dataframe(reco)
            

    if choose_song == 'Chanson Rock': 
        song = [choose_genre, np.nan, 0.021,	0.726	,0.654,	0.000000,	0.0060,	0.0917,	0.335	,0.8,	0,	0.699864,	0.839256]
                    
        #user's input
        if choose_genre == 'Rock':
            st.text("\n")
            st.image("Pictures/image graph Rock et chanson 3.png", width=780)          
            st.text("\n")
            st.text("\n")
            with st.expander('Lancez votre prédiction'):
                

                #TEsting others feature values to go between median and 3rd quartile
                #creating df based on user's input
                df_music_NN_genre_3 = df_music_NN[df_music_NN['genre_lower']==choose_genre.lower()]

                #preparing nearest neighbors
                columns = ['acousticness',	'danceability',	'energy',	'instrumentalness',	'liveness',	'speechiness',	'valence','time_sig_number', 'mode_number', 'tempo_scaled',	'loudness_scaled']
                y_NN_3 = df_music_NN_genre_3['popularity']
                X_NN_3 = df_music_NN_genre_3[columns]

                #fitting model
                model_NN_3 = NearestNeighbors(n_neighbors = 3)
                model_NN_3.fit(X_NN_3)

                #user's song's data =>lower than median
                data_NN_pop_3 = pd.DataFrame(data = [song], columns  = ['genre_lower',	'popularity',	'acousticness',	'danceability',	'energy',	'instrumentalness',	'liveness',	'speechiness',	'valence',	'time_sig_number',	'mode_number', 'tempo_scaled',	'loudness_scaled'], index = [250000] )

                #getting neighbors
                distances_3, indices_3 = model_NN_3.kneighbors(data_NN_pop_3[columns])
                print(distances_3)
                print(indices_3)

                #geting 3 nearest neighbors in df
                output_NN_3 = df_music_NN_genre_3.iloc[[indices_3[0][0], indices_3[0][1], indices_3[0][2]], :]

                #getting distance weighted populqrity
                total_dist_3 = (1/distances_3[0][0] + 1/distances_3[0][1] + 1/distances_3[0][2]) 
                output_popularity_3 = (output_NN_3['popularity'].iloc[0]/(distances_3[0][0] * total_dist_3)) + (output_NN_3['popularity'].iloc[1]/(distances_3[0][1] *total_dist_3))+ (output_NN_3['popularity'].iloc[2]/(distances_3[0][2] * total_dist_3))

                #getting all results in a beautiful dataframe
                pop_and_confidence_3 = pd.DataFrame(data = [choose_genre, distances_3[0][0], round(output_popularity_3)])
                pop_and_confidence_3 = pop_and_confidence_3.T
                pop_and_confidence_3.rename(columns = {0 : 'Genre', 1: 'Distance', 2 : 'Popularity'}, inplace = True)


                #getting confidence based on distance of first nearest neighbors
                #the function to determien which confidence we have
                def confidence(x) : 
                    distance_IC = [0.15, 0.30, 0.60, 0.9]
                    IC = ['Very good', 'Good', 'Be careful', "Something's wrong"]
                    for i in range(len(distance_IC)):
                        if x < distance_IC[i] :
                            x = IC[i]
                            return x
                    return "Don't worry We will help you"

                st.markdown("<h5 style='text-align: left; color : black;'>La popularité de votre chanson </h5>", unsafe_allow_html=True)
                st.text("\n")
                st.write("Dans le tableau ci-dessous, vous trouverez le rappel du genre sélectionné (genre), la note de popularité (popularity) que nous vous prédisons ainsi que la popularité seuil à atteindre en fonction de votre genre musical (popularity threshold). Atteindre ce seuil vous place dans les 25% de chansons les plus populaires du genre.")           
                st.text("\n")
                pop_and_confidence_3['Confidence'] = pop_and_confidence_3['Distance'].apply(confidence)
                pop_and_confidence_3['Popularity Threshold'] = df_music_NN[df_music_NN['genre_lower'] == choose_genre.lower()]['popularity'].quantile(0.75).astype(int)
                pop_and_confidence_3[['Genre','Popularity', 'Popularity Threshold']]

                for i in range(4):
                    st.text("\n")
                st.markdown("<h5 style='text-align: left; color : black;'>Et si nous travaillions ensemble votre chanson</h5>", unsafe_allow_html=True)
         
                col1, col2, col3 = st.columns([1,6,1])
                with col1:
                    st.write("")
                with col2:
                    st.image("Pictures/image2 sep module (1).jpg", width=500)
                with col3:
                    st.write("")
                
                st.text("\n")
                st.write("Pour faire progresser votre popularité, nous vous proposons de modifier les paramètres de votre chanson.")
                st.write("Pour cela, allez modifier les différents curseurs à gauche pour voir évoluer votre popularité et vous rapprocher des étoiles.")
                st.text("\n")
                st.text("\n")
                
                Flag = False
            
                if not Flag :

                    #FEATURES TUNING TEST 
                    
                    
                    #creating df based on user's input
                    df_music_NN_genre_10 = df_music_NN[df_music_NN['genre_lower']==choose_genre.lower()]

                    #preparing nearest neighbors
                    columns = ['acousticness',	'danceability',	'energy',	'instrumentalness',	'liveness',	'speechiness',	'valence','time_sig_number', 'mode_number','tempo_scaled',	'loudness_scaled']
                    y_NN_10 = df_music_NN_genre_10['popularity']
                    X_NN_10 = df_music_NN_genre_10[columns]

                    #fitting model
                    model_NN_10 = NearestNeighbors(n_neighbors = 3)
                    model_NN_10.fit(X_NN_10)

                    #user's song's data
                    # Create sidebar to tune parameters :
                    st.sidebar.markdown("## Tune your features")
                    danceability = st.sidebar.slider("Level of danceability", min_value=0.0, max_value=1.0, step=0.0100, value = 0.93)
                    acousticness = st.sidebar.slider("Level of acousticness", min_value=0.0, max_value=1.0, step=0.0100, value = 0.72)
                    tempo_scaled = st.sidebar.slider("Level of tempo", min_value=0.0, max_value=1.0, step=0.100, value = 0.8)
                    time_sig_number = st.sidebar.slider("Level of time_signature", min_value=0.0, max_value=1.0, step=0.2)
                    #danceability = 0.93
                    #acousticness = 0.72
                    #tempo_scaled = 0.8
                    #time_sig_number = 1


                    data_NN_pop_10 = pd.DataFrame(data = [[choose_genre, np.nan, acousticness,	danceability, 0.654,	0.000000,	0.0060,	0.0917,	0.335,	time_sig_number,	0, tempo_scaled,	0.839256]], 
                                                columns  = ['genre_lower',	'popularity',	'acousticness',	'danceability',	'energy',	'instrumentalness',	'liveness',	'speechiness',	'valence',	'time_sig_number',	'mode_number', 'tempo_scaled',	'loudness_scaled'], index = [250000] )

                    #getting neighbors
                    distances_10, indices_10 = model_NN_10.kneighbors(data_NN_pop_10[columns])
                    

                    #geting results
                    output_NN_10 = df_music_NN_genre_10.iloc[[indices_10[0][0], indices_10[0][1], indices_10[0][2]], :]
                    

                    #getting popularity
                    total_dist_10 = (1/distances_10[0][0] + 1/distances_10[0][1] + 1/distances_10[0][2]) 
                    output_popularity_10 = ((output_NN_10['popularity'].iloc[0]/(distances_10[0][0] * total_dist_10)) + (output_NN_10['popularity'].iloc[1]/(distances_10[0][1] *total_dist_10))+ (output_NN_10['popularity'].iloc[2]/(distances_10[0][2] * total_dist_10)))

                    #getting all results in a beautiful dataframe
                    pop_and_confidence_10 = pd.DataFrame(data = [choose_genre, distances_10[0][0], round(output_popularity_10)])
                    pop_and_confidence_10 = pop_and_confidence_10.T
                    pop_and_confidence_10.rename(columns = {0 : 'Genre', 1: 'Distance', 2 : 'Popularity'}, inplace = True)


                    #getting confidence based on distance of first nearest neighbors
                    #the function to determien which confidence we have
                    def confidence(x) : 
                        distance_IC = [0.15, 0.30, 0.60, 0.9]
                        IC = ['Very good', 'Good', 'Be careful', "Something's wrong"]
                        for i in range(len(distance_IC)):
                            if x < distance_IC[i] :
                                x = IC[i]
                                return x
                        return "Don't worry We will help you"

                    pop_and_confidence_10['Confidence'] = pop_and_confidence_10['Distance'].apply(confidence)
                    pop_and_confidence_10['Popularity Threshold'] = df_music_NN[df_music_NN['genre_lower'] == choose_genre.lower()]['popularity'].quantile(0.75).astype(int)
                    pop_and_confidence_10[['Genre','Popularity','Popularity Threshold']]

                    #texte POUR #3 LIRE LES RESULTATS
                    for i in range(3):
                        st.text("\n")

                    st.markdown("<h5 style='text-align: left; color : black;'>Votre prédiction de popularité :</h5>", unsafe_allow_html=True)
                    st.write(
                    """
                    Nous analysons pour vous les caractéristiques techniques des chansons de votre catégorie de genre. 
                    Voici quelques exemples de caractéristiques techniques analysées : 'danceability' (la possibilité de danser sur votre musique), le tempo, le time signature (signature temporelle), l'acoustiness (l'acuité acoustique), etc. 
                    Ces caractéristiques sont présentes lors de la construction de votre chanson. Vous pouvez les retrouver grâce à votre studio d'enregistrement ou grâce à un logiciel dédié.
                    """) 
                    
                    st.text("\n")

                    
                     


            
