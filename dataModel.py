# -*- coding: utf-8 -*-
# %% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#% matplotlib inline

# %% create list array
my_list=[1,2,3]
arr=np.array(my_list)
arr

# %% Import Customer data
customers = pd.read_csv('Ecommerce Customers.csv')
    
# customer info & decribe
customers.info()
customers.describe()
customers.head()

# %% joint plotting

sns.jointplot(data=customers, x='Time on App',y='Yearly Amount Spent')
sns.jointplot(data=customers, x='Time on Website',y='Yearly Amount Spent')
sns.jointplot(data=customers, x='Time on App',y='Length of Membership',kind='hex')
    
# draw pair plot
sns.pairplot(customers)

# %% linear modelling

sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=customers)
    
# Create testset to train the linear model
customers.columns
y=customers['Yearly Amount Spent']
X=customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]                                                                                           

# %% train the linearmodel

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import  LinearRegression
lm = LinearRegression()

# %%