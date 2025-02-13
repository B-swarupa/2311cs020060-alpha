#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
titanic=pd.read_csv("Titanic.csv")
titanic


# In[5]:


titanic.info()


# #### observations
# - all columns are object data type and categorical in nature
# - there are no null values
# - as the columns are categoricaal,we can adopt one-hot-encoding

# In[7]:


import matplotlib.pyplot as plt
counts=titanic["Class"].value_counts()
plt.bar(counts.index,counts.values)


# In[8]:


df=pd.get_dummies(titanic,dtype=int)
df.head()


# In[9]:


df.info()


# In[ ]:


get_ipython().system('pip install mlxtend')
import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import apriori,association_rules


# In[17]:


frequent_itemsets=apriori(df,min_support=0.05,use_colnames=True,max_len=None)
frequent_itemsets


# In[12]:


rules=association_rules(frequent_itemsets,metric="lift",min_threshold=1.0)
rules


# In[13]:


rules.sort_values(by='lift',ascending=False)


# In[14]:


rules[['support','confidence','lift',]].hist(figsize=(15,7))
plt.show()


# #### observations
# - highest  frequency range of support is
# - highest frequency range of condifidence is 1.0
# - highest  frequency range of lift is 1-1.5

# In[ ]:


plt.scatter(rules['support'],rules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


# In[ ]:


plt.scatter(rules['confidence'],rules['lift'])
plt.xlabel('confidence')
plt.ylabel('lift')
plt.show()


# In[16]:


rules[rules["consequents"]==({"Survived_Yes"})]


# In[ ]:





# In[ ]:





# In[ ]:




