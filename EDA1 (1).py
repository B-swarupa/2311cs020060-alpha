#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv("data_clean.csv")
print(data)


# In[4]:


data.info()


# In[5]:


print(data)


# In[6]:


print(type(data))
print(data.shape)
print(data.size)


# In[7]:


data1 = data.drop(['Unnamed: 0',"Temp C"], axis=1)
data1


# In[8]:


data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[9]:


[data1.duplicated(keep=False)]


# In[10]:


data1[data1.duplicated()]


# In[11]:


data1.drop_duplicates(keep='first',inplace=True)
data1


# In[12]:


data1.rename({'Solar.R':'Solar'}, axis=1,inplace=True)
data1


# In[13]:


data1.info()


# In[14]:


data1.isnull().sum()


# In[22]:


#visualize the missing values
cols=data1.columns
colors=['black','red']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar=True)


# In[24]:


median_ozone=data1['Ozone'].median()
mean_ozone=data1['Ozone'].mean()
print("Median of ozone: ",median_ozone)
print("Mean of ozone: ",mean_ozone)


# In[26]:


data1['Ozone']=data1['Ozone'].fillna(median_ozone)
data1.isnull()


# In[27]:


data1['Ozone']=data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[31]:


#replacing the missing values in solar
data1['Solar']=data1['Solar'].fillna(median_ozone)
data1.isnull().sum()


# In[37]:


#median_solar=data1['Solar'].median()
#mean_solar=data1['Solar'].mean()
#print("Median of solar: ",median_solar)
#print("Mean of solar: ",mean_solar)


# In[36]:


#data1['Solar']=data1['Solar'].fillna(mean_solar)
#data1.isnull().sum()


# In[38]:


data1.head()


# In[39]:


print(data1['Weather'].value_counts())
mode_weather=data1['Weather'].mode()[0]
print(mode_weather)


# In[40]:


data1['Weather']=data['Weather'].fillna(mode_weather)
data1.isnull().sum()


# In[ ]:




