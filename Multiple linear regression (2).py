#!/usr/bin/env python
# coding: utf-8

# #### Assumpyions in multilinear regression
# - Lineaity:the relationship between the predictors and the response is linear
# - Independence:observations are independent of each other
# - Homoscedasticity:the residuals (difference between obsderved and predicted values)exhibit constant varianvce at all levels of the predictors
# 

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
cars=pd.read_csv('Cars.csv')
cars


# In[4]:


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

# In[9]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[10]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
sns.boxplot(data=cars, x='SP', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='SP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[11]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
sns.boxplot(data=cars, x='VOL', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='VOL', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[12]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
sns.boxplot(data=cars, x='WT', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='WT', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[13]:


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

# In[14]:


cars[cars.duplicated()]


# In[15]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)



# In[16]:


cars.corr()


# #### OBSERVATIONS
# - here all the values are between -1 to 1.
# - all the diagonals are the values of 1
# - mpg of (sp,hp,vol,wt)values are negative valules.
# - highest strengths are b/w sp vs hp,vol vas mpg.
# 
# 

# In[18]:


model1=smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()
model1.summary()


# #### variability of y is explained by x i.e r^2

# #### OBERVATIONS
# - THE R SQUARED AND ADJUSTED R SQUARED VALUES ARE GOOD AND ABOUT 75% OF VARIABILITY IN Y IS EXPLAAINED BY X COLUMNS
# - THE PROBBILITY VALUE WITH RESPECT TO F STATTICS IS CLOSE TO ZERO,INDICATING THAT ALL OR SOMEOF X COLUMNS ARE SIGNIFICANT
# - THE P VALUES FOR VOLAND WT ARE HIGHER THAN 5% INDICATING  SOME INTERACTION ISSUE AMONG THEMSELVES,WHICH NEED TO BE FURTHER EXPLORED

# In[19]:


df1=pd.DataFrame()
df1['actual_y1']=cars['MPG']
df1.head()


# In[25]:


pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[27]:


from sklearn.metrics import mean_squared_error
mse=mean_squared_error(df1['actual_y1'],df1['pred_y1'])
print('MSE: ',mse)
print('RMSE: ',np.sqrt(mse))                    


# In[28]:


#checking for multicollinearity among x columns using vif method
cars.head()


# In[29]:


# Compute VIF values
rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['Hp','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# In[ ]:





# In[ ]:




