#!/usr/bin/env python
# coding: utf-8

# #### Assumpyions in multilinear regression
# - Lineaity:the relationship between the predictors and the response is linear
# - Independence:observations are independent of each other
# - Homoscedasticity:the residuals (difference between obsderved and predicted values)exhibit constant varianvce at all levels of the predictors
# 

# In[1]:


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


# In[4]:


#rearrange the col
cars=pd.DataFrame(cars,columns=['HP','VOL','SP','WT','MPG'])
cars
                  


# In[5]:


cars.head()


# #### Description of columns
# - MPG:-milege of the car(mile per gallon)
# - VOL:-volume of the car(size)
# - SP:-top speed of the car(miles per hour)
# - WT:-weight of the car(pounds)
# - HP:-horse power of the car

# EDA

# In[6]:


cars.info()


# In[7]:


#checking for missing values
cars.isna().sum()


# #### 
# - No missing values and the given datatypes of the relevant and valid

# In[8]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[9]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
sns.boxplot(data=cars, x='SP', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='SP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[10]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
sns.boxplot(data=cars, x='VOL', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='VOL', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[11]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
sns.boxplot(data=cars, x='WT', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='WT', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[12]:


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

# In[13]:


cars[cars.duplicated()]


# In[14]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)



# In[15]:


cars.corr()


# #### OBSERVATIONS
# - here all the values are between -1 to 1.
# - all the diagonals are the values of 1
# - mpg of (sp,hp,vol,wt)values are negative valules.
# - highest strengths are b/w sp vs hp,vol vas mpg.
# 
# 

# In[16]:


model1=smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()
model1.summary()


# #### variability of y is explained by x i.e r^2

# #### OBERVATIONS
# - THE R SQUARED AND ADJUSTED R SQUARED VALUES ARE GOOD AND ABOUT 75% OF VARIABILITY IN Y IS EXPLAAINED BY X COLUMNS
# - THE PROBBILITY VALUE WITH RESPECT TO F STATTICS IS CLOSE TO ZERO,INDICATING THAT ALL OR SOMEOF X COLUMNS ARE SIGNIFICANT
# - THE P VALUES FOR VOLAND WT ARE HIGHER THAN 5% INDICATING  SOME INTERACTION ISSUE AMONG THEMSELVES,WHICH NEED TO BE FURTHER EXPLORED

# In[17]:


df1=pd.DataFrame()
df1['actual_y1']=cars['MPG']
df1.head()


# In[18]:


pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[19]:


from sklearn.metrics import mean_squared_error
mse=mean_squared_error(df1['actual_y1'],df1['pred_y1'])
print('MSE: ',mse)
print('RMSE: ',np.sqrt(mse))                    


# In[20]:


#checking for multicollinearity among x columns using vif method
cars.head()


# In[21]:


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


# #### Observations
# - The ideal range of VIF values shall be between 0 to 10. However slighty higher values can be tolerated.
# - As seen from the very high VIF values for VOL and WT ,it is clear that they are prone to multicollineraity problem.
# - Hence it is declared to drop one of the column (either VOL or WT)to overcome the multicollinearity.
# - It is decieded to drop WT and retain VOL column in futher models

# In[22]:


cars1=cars.drop("WT",axis=1)
cars1.head()


# In[25]:


cars.head()


# In[26]:


cars1


# In[27]:


#build model2 on cars1 dataset
model2=smf.ols('MPG~VOL+SP+HP',data=cars1).fit()
model2.summary()


# In[29]:


df2=pd.DataFrame()
df2['actual_y2']=cars['MPG']
df2.head()


# In[37]:


# predict for the given X data columns
pred_y2 = model2.predict(cars.iloc[:,0:4])
df2["pred_y2"] = pred_y2
df2.head()


# In[36]:


from sklearn.metrics import mean_squared_error
mse=mean_squared_error(df2['actual_y2'],df2['pred_y2'])
print('MSE: ',mse)
print('RMSE: ',np.sqrt(mse))


# #### OBSERVATIONS
# - THE ADJUSTED R SQUARED VALUE IMPROVED SLIGHTLY TO 0.76
# - ALL THE P VALUES FOR MODEL PARAMETERS ARE LESS THAN 5% HENCE THEY ARE SIGNIFICANT
# - THEREFORE THE HP,VOL,SP COLUMNS ARE FINALIZED AS THE SIGNIFICANT PREDICTOR FOR THE MPG
# - THERE   IS NO IMPROVEMENT IN MSE VALUE

# ## IDENTIFICATION OF HIGH INFLUENCE POINTS

# In[39]:


# define variables and assign values
k=3#no of x columns
n=81#no of observations(rows)
leverage_cutoff=3*((k+1)/n)
leverage_cutoff


# In[40]:


influence_plot(model1,alpha=0.5)
y=[i for i  in range(-2,8)]
x=[leverage_cutoff for i in range(10)]
plt.plot(x,y,'r+')
plt.show()


# #### observations
# - red lines are known as leverage cutoffs
# - index number of record are the numbers in the datapoint
# - (outliers) or (influencation points)  are right side of the red line & also which are large in size
# - from the above plot,it is evident that data points 65,70,76,78,79,80 are the influencers
# - as their H leverage values are higher and size is higher.

# In[43]:


cars2=cars1.drop(cars1.index[[65,70,76,78,79,80]],axis=0).reset_index(drop=True)
cars2


# In[44]:


#build model3 on cars2 dataset
model3=smf.ols('MPG~VOL+SP+HP',data=cars2).fit()
model3.summary()


# In[ ]:




