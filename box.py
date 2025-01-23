#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
s1=[20,15,10,25,30,35,28,40,45,60]
scores1=pd.Series(s1)
scores1


# In[4]:


import matplotlib.pyplot as plt
plt.boxplot(scores1,vert=False)


# In[5]:


plt.figure(figsize=(6,2))
plt.title("Boxplot for batsman scores")
plt.xlabel("scores")
plt.boxplot(scores1,vert=False)


# In[6]:


s2=[20,15,10,25,30,35,28,40,45,60,120,150]
scores2=pd.Series(s2)
print(scores2)
plt.figure(figsize=(6,2))
plt.title("Boxplot for batsman scores")
plt.xlabel("scores")
plt.boxplot(scores2,vert=False)


# In[ ]:




