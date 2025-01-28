#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv("data_clean.csv")
print(data)


# In[3]:


data.info()


# In[4]:


print(data)


# In[5]:


print(type(data))
print(data.shape)
print(data.size)


# In[6]:


data1 = data.drop(['Unnamed: 0',"Temp C"], axis=1)
data1


# In[7]:


data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[8]:


[data1.duplicated(keep=False)]


# In[9]:


data1[data1.duplicated()]


# In[10]:


data1.drop_duplicates(keep='first',inplace=True)
data1


# In[11]:


data1.rename({'Solar.R':'Solar'}, axis=1,inplace=True)
data1


# In[12]:


data1.info()


# In[13]:


data1.isnull().sum()


# In[14]:


#visualize the missing values
cols=data1.columns
colors=['black','red']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar=True)


# In[15]:


median_ozone=data1['Ozone'].median()
mean_ozone=data1['Ozone'].mean()
print("Median of ozone: ",median_ozone)
print("Mean of ozone: ",mean_ozone)


# In[16]:


data1['Ozone']=data1['Ozone'].fillna(median_ozone)
data1.isnull()


# In[17]:


data1['Ozone']=data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[18]:


#replacing the missing values in solar
data1['Solar']=data1['Solar'].fillna(median_ozone)
data1.isnull().sum()


# In[19]:


#median_solar=data1['Solar'].median()
#mean_solar=data1['Solar'].mean()
#print("Median of solar: ",median_solar)
#print("Mean of solar: ",mean_solar)


# In[20]:


#data1['Solar']=data1['Solar'].fillna(mean_solar)
#data1.isnull().sum()


# In[21]:


data1.head()


# In[22]:


print(data1['Weather'].value_counts())
mode_weather=data1['Weather'].mode()[0]
print(mode_weather)


# In[23]:


data1['Weather']=data['Weather'].fillna(mode_weather)
data1.isnull().sum()


# In[24]:


print(data1['Month'].value_counts())
mode_month=data1['Month'].mode()[0]
print(mode_month)


# In[25]:


data1['Month']=data['Month'].fillna(mode_month)
data1.isnull().sum()


# In[26]:


data1.tail()


# In[27]:


data1.reset_index(drop=True)


# In[28]:


#detection of outliers in the column

# Create subplots
fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 3]})
# Boxplot
sns.boxplot(data=data1["Ozone"], ax=axes[0], color='skyblue', width=0.5, orient='h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Ozone levels")
# Histogram with KDE
sns.histplot(data1["Ozone"], kde=True, ax=axes[1], color='purple', bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Ozone levels")
axes[1].set_ylabel("Frequency")
# Adjust layout and show
plt.tight_layout()
plt.show()


# #### OBSERVATIONS
# - The ozone colum has extreme values beyond 81 as seen from box plot.
# - the same is confirmes from the below right=skewed histogram.
# - where the ozone level in histrogram with kde is more than 40 frequency.
# 

# In[29]:


fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 3]})
# Boxplot
sns.boxplot(data=data1["Solar"], ax=axes[0], color='skyblue', width=0.5, orient='h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Solar levels")
# Histogram with KDE
sns.histplot(data1["Solar"], kde=True, ax=axes[1], color='purple', bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Solar levels")
axes[1].set_ylabel("Frequency")
# Adjust layout and show
plt.tight_layout()
plt.show()


# In[30]:


#create a figure for violin plot
sns.violinplot(data=data1["Ozone"],color='lightgreen')
plt.title("violoin plot")
plt.show()


# In[49]:


#method1
plt.figure(figsize=(6,2))
boxplot_data=plt.boxplot(data1["Ozone"],vert=False)
[item.get_xdata() for item in boxplot_data['filters']]


# In[37]:


data1['Ozone'].describe()


# In[48]:


mu=data1["Ozone"].describe()[1]
sigma =data1["Ozone"].describe()[2]
for x in data1["Ozone"]:
    if ((x  <(mu - 3 *sigma)) or (x  >(mu  +3 *sigma))):
        print(x)


# In[45]:


import scipy.stats as stats
plt.figure(figsize=(8,6))
stats.probplot(data1["Ozone"],dist="norm",plot=plt)
plt.title("Q-Q plot for outlier detection",fontsize=14)
plt.xlabel("Theoretical quantiles",fontsize=12)


# In[46]:


import scipy.stats as stats
plt.figure(figsize=(8,6))
stats.probplot(data1["Solar"],dist="norm",plot=plt)
plt.title("Q-Q plot for outlier detection",fontsize=14)
plt.xlabel("Theoretical quantiles",fontsize=12)


# #### Observations from q-q plot
# - the data does not follow distribution as the data pointys are deviating significantly away from the red line
# - the data shows a right skewes distribution and possible outliers

# In[ ]:




