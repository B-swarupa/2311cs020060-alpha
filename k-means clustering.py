#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster  import KMeans
Univ = pd.read_csv("Universities.csv.clustering.csv")
Univ
                


# In[2]:


Univ.info()


# In[3]:


Univ.isna().sum()


# In[4]:


Univ.describe()


# In[5]:


Univ1 = Univ.iloc[:,1:]
Univ1


# In[6]:


cols=Univ1.columns


# In[7]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled_Univ_df=pd.DataFrame(scaler.fit_transform(Univ1),columns=cols)
scaled_Univ_df


# In[8]:


#build 3 clusters using kmeans cluster algorithm
from sklearn.cluster import KMeans
clusters_new=KMeans(3,random_state=0)
clusters_new.fit(scaled_Univ_df)


# In[9]:


#print the cluster labels
clusters_new.labels_


# In[10]:


set(clusters_new.labels_)


# In[12]:


#assign clusters to the data set
Univ['clusterid_new']=clusters_new.labels_
Univ


# In[13]:


Univ[Univ['clusterid_new']==1]


# In[14]:


Univ[Univ['clusterid_new']==0]


# In[15]:


Univ[Univ['clusterid_new']==2]


# In[16]:


#use groupby() to find aggregated(mean)values in each cluster
Univ.iloc[:,1:].groupby("clusterid_new").mean()


# #### observations
# - cluster 2 appears to be top rated universities clusteer as the cutoff score and top 10,sf ratio parameters mean values are highest
# - cluster 1 appears to occupy the middle level rated universities
# - cluster 0 scores as the lower level rated universities.
# 

# In[17]:


#### finding a optimal k value using elbow plot
wcss=[]
for i in range(1,20):
    kmeans=KMeans(n_clusters=i,random_state=0)
    kmeans.fit(scaled_Univ_df)
    wcss.append(kmeans.inertia_)
print(wcss)
plt.plot(range(1,20),wcss)
plt.title("Elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("wcss")
plt.show()
#wcss is the varience value in the cluster        


# In[ ]:




