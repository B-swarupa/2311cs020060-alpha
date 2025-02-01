#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
data1=pd.read_csv("NewspaperData.csv")
data1


# In[3]:


data1.info()


# In[4]:


data1.describe()


# In[11]:


plt.boxplot(data1["daily"])


# In[5]:


plt.scatter(data1["daily"],data1["sunday"])


# In[6]:


data1['daily'].corr(data1['sunday'])


# #### Observations
# - the relationship between x daily and y sunday is seen to be linear as seen from scatter plot
# - the corr is strong and positive with pearsons corr coefficient of 0.95

# In[7]:


model=smf.ols("sunday~daily",data=data1).fit()
model.summary()


# #### observations 
# - predicted equation is beta_0=13.8356,beta_1=1.3397 x
# - beta_0+beta_1*x=y_hat
# - The probability (p-value) for intercept(beta_0) is 0.707>0.05
# - therefore the intercept coefficinet may not be that much significant in prediction
# - however the p-value for 'daily'(beta_1) is 0.00<0.05
# - therefore the beta_1 coefficinet is highly significant and id contributint to prediction.

# In[8]:


x=data1["daily"].values
y=data1["sunday"].values
plt.scatter(x,y,color="m",marker="o",s=30)
b0=13.84
b1=1.33
y_hat=b0+b1*x
plt.plot(x,y_hat,color="g")
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[9]:


x=data1['daily']
y=data1['sunday']
plt.scatter(data1['daily'],data1['sunday'])
plt.xlim(0,max(x)+100)
plt.ylim(0,max(y)+100)
plt.show()


# In[ ]:




