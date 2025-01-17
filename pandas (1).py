#!/usr/bin/env python
# coding: utf-8

# In[3]:


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


# In[5]:


#creating dictionary
data={'a':1,'b':2,'c':3,'d':4}
series=pd.Series(data)
print(series)
#abcd values  dictionary lo keys avutayi


# In[6]:


series.replace(4,5)


# In[7]:


#creating series using numpy array
import numpy as np
data=np.array([1,2,34,5,6,6])
series=pd.Series(data,index=['a','b','c','d','e','f'])
print(series)


# In[8]:


#tablur info ni pandas lo dataframe ani antaru
#creating pandas dataframe
import pandas as pd
data={'Name':['alice','bob','mary'],'Age':[23,34,12],'Countrty':['USA','UK','IND']}
df=pd.DataFrame(data)
print(df)


# In[9]:


#creating pandas dataframe from numpay array
import numpy as np
array=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(array)
df=pd.DataFrame(array,columns=['a','b','c'])
print(df)


# In[10]:


import pandas as pd
iris_df=pd.read_csv("iris.csv")
print(iris_df)


# In[11]:


iris_df.info()


# In[12]:


iris_df.head()
#first 5 ni display chestadhi,()lo number estey dhani bati kuda display chestadhi


# In[13]:


iris_df.tail()
#last 5 ni display chestadhi,()lo number estey dhani bati kuda display chestadhi


# In[14]:


iris_df.tail(9)


# In[15]:


iris_df.describe()


# In[16]:


print(iris_df.shape)
print(iris_df.ndim)#number od dimensions
print(iris_df.size)



# In[17]:


import numpy as np
iris_array=np.array(iris_df)
iris_array


# In[18]:


iris_df


# In[19]:


iris_df[["sepal.length","petal.width"]]


# In[20]:


iris_df.iloc[1,1]


# In[21]:


iris_df.iloc[10:21, :  ]


# In[22]:


iris_df.iloc[10:20,[0,3]]
#iloc accepts numbers 


# In[23]:


iris_df.loc[10:15,"sepal.length":"petal.width"]
#loc accepts names


# In[24]:


iris_df.loc[(10,15),"sepal.length":"petal.width"]


# In[25]:


#create a dataframe with the array
import pandas as pd
data={"weight":[76,34,56,98,67,49],\
      "height":[167,187,145,189,154,178]}
bmi=pd.DataFrame(data)
bmi


# In[26]:


#create bmi column
bmi["BMI"]=bmi["weight"]/(bmi["height"]/100)**2
bmi


# In[27]:


bmi.loc[3]=[78,np.nan,np.nan]
bmi


# In[28]:


import pandas as pd
bmi.loc[5]=[np.nan,65,np.nan]
bmi


# In[46]:


bmi.info()


# In[47]:


bmi.isnull().sum()


# In[32]:


bmi["weight"]=bmi["weight"].fillna(65)
bmi


# In[33]:


bmi["height"]=bmi["height"].fillna(165)
bmi
#fillna-fill null values


# In[34]:


bmi["BMI_new"]=bmi["weight"]/(bmi["height"]/100)**2
bmi


# In[35]:


bmi.drop("BMI",axis=1)


# bmi.drop("BMI",axis=1,inplace=True)

# In[39]:


#data visualization on iris dataset
#iris_df=pd.read_csv("iris.csv")
import  matplotlib.pyplot as plt
plt.hist(iris_df["sepal.length"],color="black")


# In[ ]:




