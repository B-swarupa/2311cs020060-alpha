#!/usr/bin/env python
# coding: utf-8

# #### Assumpyions in multilinear regression
# - Lineaity:the relationship between the predictors and the response is linear
# - Independence:observations are independent of each other
# - Homoscedasticity:the residuals (difference between obsderved and predicted values)exhibit constant varianvce at all levels of the predictors
# 

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
cars=pd.read_csv('Cars.csv')
cars


# In[3]:


cars.head()


# In[5]:


#rearrange the col
cars=pd.DataFrame(cars,columns=['HP','VOL','SP','WT','MPG'])
cars
                  


# In[6]:


cars.head()


# #### Description of columns
# - MPG:-milege of the car(mile per gallon)
# - VOL:-volume of the car(size)
# - SP:-top speed of the car(miles per hour)
# - WT:-weight of the car(pounds)
# - HP:-horse power of the car

# EDA

# In[7]:


cars.info()


# In[8]:


#checking for missing values
cars.isna().sum()


# #### 
# - No missing values and the given datatypes of the relevant and valid

# In[11]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[ ]:




