#!/usr/bin/env python
# coding: utf-8

# In[1]:


#lambda functions
greet=lambda name:print(f"good job {name}!")
greet('swarupa')


# In[11]:


value=lambda a,b,c : a*b*c

value(1,2,4)


# In[14]:


#lambda function with list comprehension

even = lambda L:[x for x in L if x%2==0]
my_list=[78,96,12,98,56,7,3]
even(my_list)


# In[ ]:




