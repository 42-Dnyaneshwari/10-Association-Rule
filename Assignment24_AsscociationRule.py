# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:32:25 2023

@author: Dnyaneshwari
"""

"""
Data Dictionary:- 
Data Dictionary:- 
Feature         Description         Type          Quantitative,Nominal

Buisness Objective:- To develop a medicine effective for all types of patients
Buisness Constraint:- 

"""
###############################################################################
#Importing Libraries, Files
import pandas as pd
df = pd.read_csv('C:/Datasets/Book.csv')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans 
from scipy.linalg import svd
from sklearn.decomposition import TruncatedSVD 
from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch 
###############################################################################

df.dtypes
"""
ChildBks     int64
YouthBks     int64
CookBks      int64
DoItYBks     int64
RefBks       int64
ArtBks       int64
GeogBks      int64
ItalCook     int64
ItalAtlas    int64
ItalArt      int64
Florence     int64
dtype: object
"""
df.columns
"""
Index(['ChildBks', 'YouthBks', 'CookBks', 'DoItYBks', 'RefBks', 'ArtBks',
       'GeogBks', 'ItalCook', 'ItalAtlas', 'ItalArt', 'Florence'],
      dtype='object')

13 features
""" 
df.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2000 entries, 0 to 1999
Data columns (total 11 columns):
 #   Column     Non-Null Count  Dtype
---  ------     --------------  -----
 0   ChildBks   2000 non-null   int64
 1   YouthBks   2000 non-null   int64
 2   CookBks    2000 non-null   int64
 3   DoItYBks   2000 non-null   int64
 4   RefBks     2000 non-null   int64
 5   ArtBks     2000 non-null   int64
 6   GeogBks    2000 non-null   int64
 7   ItalCook   2000 non-null   int64
 8   ItalAtlas  2000 non-null   int64
 9   ItalArt    2000 non-null   int64
 10  Florence   2000 non-null   int64
dtypes: int64(11)
memory usage: 172.0 KB  
'''
"""Here we can see that only oldpeak is in float64 while others are in int64, we won't change the data type here"""

df.shape
#(2000, 11)

#Now let's Check for the null values
df.isnull().sum()
'''
ChildBks     0
YouthBks     0
CookBks      0
DoItYBks     0
RefBks       0
ArtBks       0
GeogBks      0
ItalCook     0
ItalAtlas    0
ItalArt      0
Florence     0
dtype: int64
'''

#So we are having no null value in any of the features

df.describe()
"""
          ChildBks     YouthBks  ...      ItalArt     Florence
count  2000.000000  2000.000000  ...  2000.000000  2000.000000
mean      0.423000     0.247500  ...     0.048500     0.108500
std       0.494159     0.431668  ...     0.214874     0.311089
min       0.000000     0.000000  ...     0.000000     0.000000
25%       0.000000     0.000000  ...     0.000000     0.000000
50%       0.000000     0.000000  ...     0.000000     0.000000
75%       1.000000     0.000000  ...     0.000000     0.000000
max       1.000000     1.000000  ...     1.000000     1.000000

"""
df['YouthBks'].value_counts()
#0    1505
#1     495
#Name: YouthBks, dtype: int64


df['ChildBks'].value_counts()
#0    1154
#1     846
#Name: ChildBks, dtype: int64

df['CookBks'].value_counts()

df['DoItYBks'].value_counts()

df['RefBks'].value_counts()

df['ArtBks'].value_counts()

df['GeogBks'].value_counts()

df['ItalCook'].value_counts()

df['ItalAtlas'].value_counts()

df['ItalArt'].value_counts()

df['Florence'].value_counts()


#So the most of the patients does not feel chest pain.
plt.rcParams['figure.figsize'] = (12,6)
sns.boxplot(df)
#Here we can see that we are having outliers in some features and we need to treat them 
#but before that we will find the correlation between the features so that we can reduce the 
#number of features by removing the uneccesary features
#So for these we will plot the heatmap
sns.heatmap(df.corr(),cmap='coolwarm', annot=True, fmt='.2f')
#We eill select the featurs which are having the positive corelation among each othr for furrther analysis i.e >0.2
#CookBks
#DoItYBks
#GeogBks
#ChildBks


#i.i we will select age,ca,oldpeak,chol,trestbps,cp,thalach,exang
df_new = df[['ChildBks','GeogBks','DoItYBks','CookBks']]
df_new

sns.boxplot(df_new)
#######################################################################
#Now let's perform univariate analysis
#age
df_new['ChildBks'].describe()
#From this it is clear that it is not having any oulier 
#the avg age is 54
#the youngest patient is 29 yr old
#the oldest one is 77 yr old 
sns.boxplot(df_new['ChildBks'])

sns.histplot(df_new['ChildBks'],kde=True)
#This is a normal curve with no outlier 
#=======================================================================
#cp
df_new['CookBks'].describe()

sns.boxplot(df_new['CookBks'])

sns.histplot(df_new['CookBks'],kde=True)
#=========================================================================
#DoItYBks
sns.boxplot(df_new['DoItYBks'])
#Here we can see some outliers letss remove those outliers by treating them with winsorization technique
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['DoItYBks']
                  )

df_new['DoItYBks'] = winsor.fit_transform(df_new[['DoItYBks']])
sns.boxplot(df_new['DoItYBks'])
#Now we cann see thaat the outliers havbeen removed

sns.histplot(df_new['DoItYBks'],kde=True)
#Also we can see that it has become a normal curve.
#========================================================================
#chol
sns.boxplot(df_new['GeogBks'])
#Here we can see some outliers letss remove those outliers by treating them with winsorization technique
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['GeogBks']
                  )
df_new['GeogBks'] = winsor.fit_transform(df_new[['GeogBks']])

sns.boxplot(df_new['GeogBks'])
#Now we cann see thaat the outliers havbeen removed
sns.histplot(df_new['GeogBks'],kde=True)
#Also we can see that it has become a normal curve.
#========================================================================
#######################################
#Hierarchical Clustering
z=linkage(df_new, method='complete',metric='euclidean') 
plt.figure(figsize=(15,8))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
sch.dendrogram(z,leaf_rotation=90,leaf_font_size=10)
plt.show()

from sklearn.cluster import AgglomerativeClustering 
h_complete = AgglomerativeClustering(n_clusters=3,linkage='complete',affinity='euclidean').fit(df_new)
#apply labels to the clusters 
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_) 
df_new['Clust'] =  cluster_labels
df_new.columns
df_final = df_new.loc[:,['Clust','ChildBks', 'GeogBks', 'DoItYBks', 'CookBks']]
df_final.iloc[:,1:].groupby(df_new.Clust).mean()
df_final.to_csv('C:\Datasets\sample.csv')
#==============================================================================
#K-Means Clustering 
TWSS = []
k = list(range(2,8))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_new)
    
    TWSS.append(kmeans.inertia_)#total within sum of square 
   
TWSS
#As k value increases the TWSS value decreases 
plt.plot(k,TWSS,'ro-');
plt.xlabel('No_of_clusters');
plt.ylabel('Total_within_SS');
#From the Scree Plot we can decide that 3 cluster is the best possible .
model = KMeans(n_clusters=3)    
model.fit(df_final)    
model.labels_ #This shows that the data point is in which cluster   
mb = pd.Series(model.labels_)    
df_final['Clust'] = mb    
df_final.head()  
df_final.columns  
df_final = df_new.loc[:,['Clust','ChildBks', 'GeogBks', 'DoItYBks', 'CookBks']]
df_final.iloc[:,1:].groupby(df_new.Clust).mean()
df_final.to_csv('C:\Datasets\K-means.csv',encoding='utf-8')

#===============================================================================
#Performing PCA 
df_ = df_new.drop({'Clust'},axis=1)
U,d,Vt = svd(df_)
svd = TruncatedSVD(n_components=3)
svd.fit(df_)
result = pd.DataFrame(svd.transform(df_))
result.head()
result.columns="pc0","pc1","pc2"
result.head()

#Scatter Diagram 
plt.scatter(x=result.pc0, y=result.pc1)

##################################################################

#Association Rules












