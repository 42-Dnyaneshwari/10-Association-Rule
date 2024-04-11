# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 08:23:53 2023

@author: Dnyaneshwari...
"""
#   RECOMMENDATION SYSTEM
import pandas as pd
anime=pd.read_csv("C:/Datasets/anime.csv",encoding='utf-8')
anime
anime.shape
#(12294, 7)

anime.columns
anime.genre
#considering only genre column

#TfidfVectorizer 
#how many tomes a perticular term is repeating itself.
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer(stop_words='english')
#it is going to create a vector which will be going 
#to seperate the words
#out of all words from the row

#checking is threre is any null value present or not
anime['genre'].isnull().sum()
#62

#replacing null value with genral
anime['genre']=anime['genre'].fillna('genral')

#Now let us create tfidf_matrix 
tfidf_matrix=tfidf.fit_transform(anime.genre)
tfidf_matrix.shape
#(12294, 47)
#it has created sparse matrix


###################################################################

#To measure the similarities
from sklearn.metrics.pairwise import linear_kernel

#because we have compare the matrix itself inorder to get similaries
cosinem_sim_mtx=linear_kernel(tfidf_matrix,tfidf_matrix)

cosinem_sim_mtx.size
#151142436
#we converting anime index to series format 
#we want index and corrsonds to
anime_index=pd.Series(anime.index,index=anime['name'].drop_duplicates)

anime_id=anime_index['Assassins (1995)']
anime_id

######################################################################


def get_rec(Name,topN):
    #topN=10
    #Name='Assassins (1995)'
    anime_id=anime_index[Name]
    #we want to capture the while row of a given movie
    #Name,Score an dits column id.
    #enumarete funtion creates a object
    ##what enumrate does
    #eg we given is (2,10,15,18,)
    #then it will give us (0,2   1,10,  2,15  ,3,18)
    cosine_score=list(enumerate(cosinem_sim_mtx[anime.Name]))
    cosine_score=sorted(cosine_score,key=lambda x:x[1], reverse=True)
    
    cosine_score_N=cosine_score[1,topN+1]
    #getting the movie index
    anime_idx=[i[0] for i in cosine_score_N]
    #getting theccosine score    
    anime_scores=[i[1] for i in cosine_score_N]   
    #a empty dataframe is created 
    anime_similar_show=pd.DataFrame(columns={['name','score']})
    anime_similar_show['name']=anime.loc[anime_idx,'name']
    #storing the index , name , score
    anime_similar_show['name']=anime.loc[anime_idx,'name']
    anime_similar_show['score']=anime_scores
    
    anime_similar_show.reset_index(inplace=True)
    print(anime_similar_show)
get_rec('Bad Boys (1995)',topN=10)    
#######################################################################

#AI
#krip