#!/usr/bin/env python
# coding: utf-8

# In[2]:


#featurs of pandas
#data structures,data cleaning,indexing and slicing,data aggregation,data transformation,time series handling,data visualization,file i/o,integration
import pandas as pd
data=[1,2,34,5,6,6]
series=pd.Series(data)
print(series)


# In[4]:


#custom indexing
data=[1,2,34,5,6,6]
i=['a','b','c','d','e','f']
series=pd.Series(data,index=i)
print(series)


# In[5]:


#pandas Series objects are size immutable but it allows to modify element value,only allows homogenues datatype


# In[6]:


#creating dictionary
data={'a':1,'b':2,'c':3,'d':4}
series=pd.Series(data)
print(series)
#abcd values  dictionary lo keys avutayi


# In[7]:


series.replace(4,5)


# In[10]:


#creating series using numpy array
import numpy as np
data=np.array([1,2,34,5,6,6])
series=pd.Series(data,index=['a','b','c','d','e','f'])
print(series)


# In[14]:


#tablur info ni pandas lo dataframe ani antaru
#creating pandas dataframe
import pandas as pd
data={'Name':['alice','bob','mary'],'Age':[23,34,12],'Countrty':['USA','UK','IND']}
df=pd.DataFrame(data)
print(df)


# In[16]:


#creating pandas dataframe from numpay array
import numpy as np
array=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(array)
df=pd.DataFrame(array,columns=['a','b','c'])
print(df)


# In[ ]:





# In[ ]:




