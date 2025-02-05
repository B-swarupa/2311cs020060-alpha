#!/usr/bin/env python
# coding: utf-8

# #### Assumpyions in multilinear regression
# - Lineaity:the relationship between the predictors and the response is linear
# - Independence:observations are independent of each other
# - Homoscedasticity:the residuals (difference between obsderved and predicted values)exhibit constant varianvce at all levels of the predictors
# 

# In[5]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
cars=pd.read_csv('Cars.csv')
cars


# In[6]:


cars.head()


# In[7]:


#rearrange the col
cars=pd.DataFrame(cars,columns=['HP','VOL','SP','WT','MPG'])
cars
                  


# In[8]:


cars.head()


# #### Description of columns
# - MPG:-milege of the car(mile per gallon)
# - VOL:-volume of the car(size)
# - SP:-top speed of the car(miles per hour)
# - WT:-weight of the car(pounds)
# - HP:-horse power of the car

# EDA

# In[9]:


cars.info()


# In[10]:


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


# In[12]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
sns.boxplot(data=cars, x='SP', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='SP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[13]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
sns.boxplot(data=cars, x='VOL', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='VOL', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[14]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
sns.boxplot(data=cars, x='WT', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='WT', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[15]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
sns.boxplot(data=cars, x='MPG', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='MPG', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# ####  OBSERVATIONS
# - these are some extreme values(outliers)observed in towards the right tail of sp and hp distributions
# - in vol and wt columns,a few outliers are observed in both tails of their distributions
# - the ertreme values of cars data may have come from the specially designed nature of cars
# - as this is multidimensional data,the outliers with respecty to spatial dimensions may have to be  considered while building the regression model.

# In[17]:


cars[cars.duplicated()]


# In[22]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)



# In[18]:


cars.corr()


# #### OBSERVATIONS
# - here all the values are between -1 to 1.
# - all the diagonals are the values of 1
# - mpg of (sp,hp,vol,wt)values are negative valules.
# - highest strengths are b/w sp vs hp,vol vas mpg.
# 
# 

# In[27]:


model1=smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()
model1.summary()


# #### variability of y is explained by x i.e r^2

# In[ ]:




