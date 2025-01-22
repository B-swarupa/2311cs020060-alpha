#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
df=pd.read_csv("universities.csv")
df


# In[3]:


np.mean(df["SAT"])


# In[4]:


np.median(df["SAT"])


# In[ ]:


#mean value in an data is heavily influenced by the outliers 
#where median is not that much effected


# In[5]:


np.std(df["GradRate"])


# In[6]:


np.var(df["SFRatio"])


# In[11]:


df.describe()


# In[8]:


#visualizise the gradrate using histogram
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(6,3))
plt.title("Acceptance Ratio")
plt.hist(df["Accept"])


# In[9]:


sns.histplot(df["Accept"],kde=True)


# In[ ]:


#in acceptance ratio the data distribution is non symmertical and right skewed.

