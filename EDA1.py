#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv("data_clean.csv")
print(data)


# In[13]:


data.info()


# In[16]:


print(data)


# In[14]:


print(type(data))
print(data.shape)
print(data.size)


# In[20]:


data1 = data.drop(['Unnamed: 0',"Temp C"], axis=1)
data1


# In[21]:


data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[23]:


[data1.duplicated(keep=False)]


# In[25]:


data1[data1.duplicated()]


# In[30]:


data1.drop_duplicates(keep='first',inplace=True)
data1


# In[ ]:




