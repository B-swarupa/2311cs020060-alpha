#!/usr/bin/env python
# coding: utf-8

# In[5]:


#offers powerful libraries
import numpy as np
x = np.array([12,56,23,98])
print(x)
print(type(x))
print(x.dtype)


# In[6]:


import numpy as np
x = np.array([12,56,23,98.90])
print(x)
print(type(x))
print(x.dtype)


# In[7]:


#verify the datatype using a character
import numpy as np
x = np.array(['A',12,56,23,98.90])
print(x)
print(type(x))
print(x.dtype)


# In[14]:


import numpy as np
x = np.array([["swaru",12,56,23,98.90],[24,45,676,89,"reddy"]])
print(x)
print(type(x))
print(x.shape)


# In[19]:


#reshaping an array
x=np.array([23,90,45,1,3,87])
y=x.reshape(2,3)
print(y.reshape)
print(y)


# In[20]:


#arange()
e=np.arange(3)
print(e)
print(type(e))
#(3)estey 0,1,2 elements print avutayi
#(1,7) antea 1,2,3,4,5,6 elements print avutayi


# In[21]:


#around()
w=np.array([5.6654,3.8657,0.7547])
print(w)
np.around(w,3)
#used for pointing the digits after point


# In[29]:


#sqrt()
w=np.array([5.6654,3.8657,0.7547])
print(w)
print(np.around(np.sqrt(w),2))


# In[31]:


a1=np.array([[5,7,9,4],[5,0,2,1]])
print(a1)
a1.dtype


# In[34]:


#astype() to convert datatype
a1_copy =a1.astype(str)
print(a1_copy)
a1_copy.dtype


# In[35]:


#mathematical operations
a2=np.array([[5,8,0],[2,9,6],[0,6,4]])
a2


# In[38]:


print(a2.sum(axis=0))
print(a2.sum(axis=1))


# In[39]:


#mean 
print(a2)
print(a2.mean(axis=0))
print(a2.mean(axis=1))


# In[41]:


#matrix 
a=np.array([[5,8,0],[2,9,6],[0,6,4]])
print(a)
np.fill_diagonal(a,0)
print(a)


# In[44]:


s=np.array([[1,2],[3,4]])
w=np.array([[5,6],[7,8]])
a=np.matmul(s,w)
print(a)
#print(s,T)
#print(w,T)


# In[46]:


#transpose
print(s.T)
print(w.T)


# In[47]:


#accessing the  array

a1=np.array([[3,4,5],[6,7,8],[9,0,1]])
a1


# In[48]:


a1[1:3,0:2]


# In[ ]:




