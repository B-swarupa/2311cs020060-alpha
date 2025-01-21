#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
df=pd.read_csv("universities.csv")
df


# In[ ]:





# In[6]:


np.mean(df["SAT"])


# In[7]:


np.median(df["SAT"])


# In[ ]:


#mean value in an data is heavily influenced by the outliers 
#where median is not that much effected


# In[9]:


np.std(df["GradRate"])


# In[10]:


np.var(df["SFRatio"])


# In[11]:


df.describe()


# In[ ]:





# In[ ]:




