#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


dataset = pd.read_csv('C:/Users/alexi/Desktop/PA/salary_data.csv')


# In[3]:


dataset.head


# In[4]:


dataset.describe()


# In[5]:


X = dataset['YearsExperience'].values.reshape (-1,1)
y = dataset['Salary'].values.reshape (-1,1)

# X = dataset.iloc[:, :-1].values #get a copy of dataset exclude last column
# y = dataset.iloc[:, 1].values #get array of dataset in column 1st


# In[6]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)


# In[7]:


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[8]:


# Visualizing the Training set results
viz_train = plt
viz_train.scatter(X_train, y_train, color='red')
viz_train.plot(X_train, regressor.predict(X_train), color='blue')
viz_train.title('Salary VS Experience (Training set)')
viz_train.xlabel('Year of Experience')
viz_train.ylabel('Salary')
viz_train.show()

# Visualizing the Test set results
viz_test = plt
viz_test.scatter(X_test, y_test, color='red')
viz_test.plot(X_train, regressor.predict(X_train), color='blue')
viz_test.title('Salary VS Experience (Test set)')
viz_test.xlabel('Year of Experience')
viz_test.ylabel('Salary')
viz_test.show()


# In[9]:


y_pred = regressor.predict([[5.0]])
print(y_pred)


# In[13]:


regressor.score(X_test, y_test)


# In[ ]:




