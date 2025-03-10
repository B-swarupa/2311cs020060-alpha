#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
movie=pd.read_csv("Movie.data.csv")
movie


# In[2]:


movie.info()


# In[3]:


import matplotlib.pyplot as plt
counts=movie["rating"].value_counts()
plt.bar(counts.index,counts.values)


# In[4]:


import matplotlib.pyplot as plt
counts=movie["movie"].value_counts()
plt.bar(counts.index,counts.values)


# In[ ]:




