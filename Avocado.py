# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 20:10:40 2022

@author: 91812
"""
# Target of this project is to predict the future price of avocados depending on those variables we have;

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

df=pd.read_csv(r'C:\Users\91812\AppData\Local\Temp\Temp6_avocado.csv.zip\avocado.csv')
df.head()
df.isnull().sum() #no null values

df.describe().round(2) #2 means upto 2 decimal values

df.info() 
#9 attributes of float values
#2 attributes of integer values
#3 attributes of categorical or object values

#now drop unnamed columns & redefine undefined colunms
df=df.drop(["Unnamed: 0"],axis=1)
df=df.rename(index=str,columns={"4046" : "Small Hass", "4225" : "Large Hass","4770" : "XLarge Hass" })

df['Date'] =pd.to_datetime(df.Date)
df.sort_values(by=['Date'])

# Average price of Conventional Avocados over time

mask = df['type']== 'conventional'
plt.rc('figure', titlesize=50)
fig = plt.figure(figsize = (26, 7))
fig.suptitle('Average Price of Conventional Avocados Over Time', fontsize=25)
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.93)

dates = df[mask]['Date'].tolist()
avgPrices = df[mask]['AveragePrice'].tolist()

plt.scatter(dates, avgPrices, c=avgPrices, cmap='plasma')
ax.set_xlabel('Date',fontsize = 15)
ax.set_ylabel('Average Price (USD)', fontsize = 15)
plt.show()

# Average price of Organic Avocados over time
mask = df['type']== 'organic'
plt.rc('figure', titlesize=50)
fig = plt.figure(figsize = (26, 7))
fig.suptitle('Average Price of Organic Avocados Over Time', fontsize=25)
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.93)

dates = df[mask]['Date'].tolist()
avgPrices = df[mask]['AveragePrice'].tolist()

plt.scatter(dates, avgPrices, c=avgPrices, cmap='plasma')
ax.set_xlabel('Date',fontsize = 15)
ax.set_ylabel('Average Price (USD)', fontsize = 15)
plt.show()



# Now, Let's see what we get from Machine Learning Algorithms!
# Dropping the Date column (date format is not suitable for next level analysis (i.e.OneHotEncoder))
df = df.drop(['Date'], axis = 1)

# Checking if the sample is balanced;
df.groupby('region').size() #almost from every region we have 338 datasets

# There are 54 regions but some are subsets of the other regions, i.e: San Francisco-California 
df.region.unique() 
len(df.region.unique()) #54

# basically we can remove states and work on cities rather than analysing both (to prevent multicollinerarity)

regionsToRemove = ['California', 'GreatLakes', 'Midsouth', 'NewYork', 'Northeast', 'SouthCarolina', 'Plains', 'SouthCentral', 'Southeast', 'TotalUS', 'West']
df = df[~df.region.isin(regionsToRemove)]
len(df.region.unique()) #now 43

#Average prices by region
plt.title("Avg.Price of Avocado by Region")
plt.figure(figsize=(10,11))
Av= sns.barplot(x="AveragePrice",y="region",data= df)

#Checking the balance in data of the types of avocados
type_counts = df.groupby('type').size()
type_counts #fairly balanced 

# The average prices of avocados by types; organic or not
plt.figure(figsize=(5,7))
plt.title("Avg.Price of Avocados by Type")
Av= sns.barplot(data=df,x="type",y="AveragePrice")

# Total Bags = Small Bags + Large Bags + XLarge Bags
# To avoid multicollinearity I'll keep S-L-XL bags and drop Total Bags

# But before droping we'd better to see the correlation between those columns:
df[['Small Hass', "Large Hass", "XLarge Hass",'Small Bags','Large Bags','XLarge Bags','Total Volume','Total Bags']].corr()

#plotting heat map
plt.figure(figsize=(12,6))
sns.heatmap(df.corr(),cmap='coolwarm',annot=True)

#Small Hass avocados are the most preferred/sold type in the US and customers tend to buy those avocados as bulk, not bag.
#Small bags has very high coreletion with total volume that means most of the volume(most sales) is from small bags

#dropping these attributes for some visualization
df_V = df.drop(['AveragePrice', 'Total Volume', 'Total Bags'], axis = 1).groupby('year').agg('sum')
df_V

indexes = ['Small Hass', 'Large Hass', 'XLarge Hass', 'Small Bags', 'Large Bags', 'XLarge Bags']
series = pd.DataFrame({'2015': df_V.loc[[2015],:].values.tolist()[0],
                      '2016': df_V.loc[[2016],:].values.tolist()[0],
                      '2017': df_V.loc[[2017],:].values.tolist()[0],
                      '2018': df_V.loc[[2018],:].values.tolist()[0]}, index=indexes)

#Some Visualization based on YEAR

#for 2015
series.plot.pie(y='2015',figsize=(9, 9), autopct='%1.1f%%', colors=['silver', 'pink', 'orange', 'palegreen', 'aqua', 'blue'], fontsize=18, legend=False, title='2015 Volume Distribution').set_ylabel('')
#for 2016
series.plot.pie(y='2016',figsize=(9, 9), autopct='%1.1f%%', colors=['silver', 'pink', 'orange', 'palegreen', 'aqua', 'blue'], fontsize=18, legend=False, title='2016 Volume Distribution').set_ylabel('')
#for 2017
series.plot.pie(y='2017',figsize=(9, 9), autopct='%1.1f%%', colors=['silver', 'pink', 'orange', 'palegreen', 'aqua', 'blue'], fontsize=18, legend=False, title='2017 Volume Distribution').set_ylabel('')
#for 2018
series.plot.pie(y='2018',figsize=(9, 9), autopct='%1.1f%%', colors=['silver', 'pink', 'orange', 'palegreen', 'aqua', 'blue'], fontsize=18, legend=False, title='2017 Volume Distribution').set_ylabel('')

# Total Bags = Small Bags + Large Bags + XLarge Bags
df = df.drop(['Total Bags'], axis = 1)

# Total Volume = Small Hass +Large Hass +XLarge Hass + Total Bags , to avoid multicollinearity I also drop Total Volume column.
df = df.drop(['Total Volume'], axis = 1)

df.info()

correlations = df.corr(method='pearson')
correlations

#Feature Scaling (Standardization).
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

df.loc[:,'Small Hass':'XLarge Bags']= scaler.fit_transform(df.loc[:,'Small Hass':'XLarge Bags']) 
df.head() 

#D.V & I.V splitting
X=df.drop(['AveragePrice'],axis=1)
Y=df['AveragePrice']

#Labeling the Categorical Variables
Xcat=pd.get_dummies(X[["type","region"]],drop_first=True)
Xnum=X[["Small Hass","Large Hass","XLarge Hass","Small Bags","Large Bags","XLarge Bags"]]

#combining categorical & numerical attributes of DV
X= pd.concat([Xcat, Xnum], axis = 1) # Concatenate dummy categorcal variables and numeric variables
X.shape

#combining DV & IV
F_DF = pd.concat([Y,X],axis=1)
F_DF.head()



#Train & Test split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30,random_state=0)



#MACHINE LEARNING ALGORITHMS

# Multiple Linear Regression
from sklearn.linear_model import LinearRegression
LinReg=LinearRegression()
LinReg.fit(X_train,Y_train)

LinReg.score(X_train,Y_train) 
print('MAE: ',metrics.mean_absolute_error(Y_test, LinReg.predict(X_test)))

# Creating a Histogram of Residuals
plt.figure(figsize=(6,4))
sns.distplot(Y_test - LinReg.predict(X_test))
plt.title('Distribution of residuals')
#We got positive or zero skew



# LASSO and RIDGE Regressions 
from sklearn import linear_model
from math import sqrt

ridge = linear_model.Ridge() 
ridge.fit(X_train, Y_train)
print('RMSE value of the Ridge Model is: ',np.sqrt(metrics.mean_squared_error(Y_test, ridge.predict(X_test))))
ridge.score(X_train, Y_train)

# Creating a Histogram of Residuals
plt.figure(figsize=(6,4))
sns.distplot(Y_test - ridge.predict(X_test))
plt.title('Distribution of residuals');

from sklearn import linear_model
lasso = linear_model.Lasso()
lasso.fit(X_train,Y_train)
lasso.score(X_train,Y_train)
print('RMSE value of the Lasso Model is: ',np.sqrt(metrics.mean_squared_error(Y_test, lasso.predict(X_test))))


# Creating a Histogram of Residuals
plt.figure(figsize=(6,4))
sns.distplot(Y_test - lasso.predict(X_test))
plt.title('Distribution of residuals');

# KNN Regressor
from sklearn import neighbors
Knn = neighbors.KNeighborsRegressor()
Knn.fit(X_train,Y_train) 
Knn.score(X_train,Y_train)

# SVR Regressor
from sklearn.svm import SVR
Svr=SVR(kernel='rbf', C=1, gamma= 0.5) #Parameter tuning to get best accuracy
Svr.fit(X_train,Y_train)
Svr.score(X_train,Y_train)

# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
DTree=DecisionTreeRegressor()
DTree.fit(X_train,Y_train)
DTree.score(X_train,Y_train)

#7- Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
RForest = RandomForestRegressor()
RForest.fit(X_train,Y_train)
RForest.score(X_train,Y_train)

# CONCLUSION 

# Comparing The RMSE Values Of The Models


# Linear Regression RMSE : 
print('RMSE value of the Linear Regr : ',round(np.sqrt(metrics.mean_squared_error(Y_test, LinReg.predict(X_test))),4))

# Ridge RMSE             : 
print('RMSE value of the Ridge Model : ',round(np.sqrt(metrics.mean_squared_error(Y_test, ridge.predict(X_test))),4))

# Lasso RMSE             : 
print('RMSE value of the Lasso Model : ',round(np.sqrt(metrics.mean_squared_error(Y_test, lasso.predict(X_test))),4))

# KNN RMSE               : 
print('RMSE value of the KNN Model   : ',round(np.sqrt(metrics.mean_squared_error(Y_test, Knn.predict(X_test))),4))

# SVR RMSE               : 
print('RMSE value of the SVR Model   : ',round(np.sqrt(metrics.mean_squared_error(Y_test, Svr.predict(X_test))),4))

# Decision Tree RMSE     : 
print('RMSE value of the Decis Tree  : ',round(np.sqrt(metrics.mean_squared_error(Y_test, DTree.predict(X_test))),4))

# Random Forest RMSE     : 
print('RMSE value of the Rnd Forest  : ',round(np.sqrt(metrics.mean_squared_error(Y_test, RForest.predict(X_test))),4))
