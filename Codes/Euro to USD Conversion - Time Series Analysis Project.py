#!/usr/bin/env python
# coding: utf-8

# # Euro to USD Conversion - Time Series Analysis Project

# ***
# _**Importing the required libraries & packages**_

# In[1]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from pandas.plotting import autocorrelation_plot
from math import sqrt
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import pickle
import warnings
warnings.filterwarnings('ignore')


# _**Changing The Default Working Directory Path & Reading the Dataset using Pandas Command and displaying first five records of the dataset**_

# In[2]:


os.chdir('C:\\Users\\Shridhar\\Desktop\\Project')
df = pd.read_csv('BOE-XUDLERD.csv')
df.head()


# ## Exploratory Data Analysis(EDA)

# _**Getting the shape of the dataset**_

# In[3]:


df.shape


# _**Checking for the null values in the dataset**_

# In[4]:


df.isna().sum()


# _**Checkiong for the duplicate values in the dataset**_

# In[5]:


df.duplicated().sum()


# _**Checking the data type of the columns in the dataset**_

# In[6]:


df.dtypes


# _**Getting the summary of various descriptive statistics for the numeric column in the dataset**_

# In[7]:


df.describe()


# ## Data Preprocessing

# _**Changing the data type of `Date` column as <span style="color:red">DateTime</span> using pandas command**_

# In[8]:


df['Date'] = pd.to_datetime(df['Date'])
df.dtypes


# _**Changing the `Date` column as the index column of the dataset and also displaying the first five records of the dataset**_

# In[9]:


df = df.set_index('Date')
df.head()


# ## Data Processing & Visualization

# _**Plotting the line graph to show the data trend in the dataset and saving the graph as PNG file**_

# In[10]:


df.plot(figsize = (10,5))
plt.title('Foreign Exchane Rate - Euro to USD')
plt.savefig('Foreign Exchane Rate - Euro to USD.png')
plt.show()


# _**Resampling the dataset to weekly frequency since the given data has more number of observations that takes longer time to fit or predict the model. And along with that showing number of observations after resampling and displaying the first five records of the resampled dataset**_

# In[11]:


df_week = df.resample('W').mean()
print('Counts of the Weekly DataFrame : ',df_week.shape[0])
df_week.head()


# _**Plotting the line graph to show the data trend in the weekly resampled dataset and saving the graph as PNG file**_

# In[12]:


df_week.plot(figsize = (10,5))
plt.title('Foreign Exchane Rate(Weekly) - Euro to USD')
plt.savefig('Foreign Exchane Rate(Weekly) - Euro to USD.png')
plt.show()


# _**Resampling the dataset to monthly frequency since the given data has more number of observations that takes longer time to fit or predict the model. And along with that showing number of observations after resampling and displaying the first five records of the resampled dataset**_

# In[13]:


df_month = df.resample('M').mean()
print('Counts of the Monthly DataFrame : ',df_month.shape[0])
df_month.head()


# _**Plotting the line graph to show the data trend in the monthly resampled dataset and saving the graph as PNG file**_

# In[14]:


df_month.plot(figsize = (10,5))
plt.title('Foreign Exchane Rate(Monthly) - Euro to USD')
plt.savefig('Foreign Exchane Rate(Monthly) - Euro to USD.png')
plt.show()


# _**Resampling the dataset to Yearly frequency since the given data has more number of observations that takes longer time to fit or predict the model. And along with that showing number of observations after resampling and displaying the first five records of the resampled dataset**_

# In[15]:


df_year = df.resample('Y').mean()
print('Counts of the Yearly DataFrame : ',df_year.shape[0])
df_year.head()


# _**Plotting the line graph to show the data trend in the Yearly resampled dataset and saving the graph as PNG file**_

# In[16]:


df_year.plot(figsize = (10,5))
plt.title('Foreign Exchane Rate(Yearly) - Euro to USD')
plt.savefig('Foreign Exchane Rate(Yearly) - Euro to USD.png')
plt.show()


# _**Plotting the scatter plot to show the data trend in the weekly resampled dataset and saving the graph as PNG file. And by observing all the above resampled graph, Weekly resampled data has more clear peaks and perks among all resample data**_

# In[17]:


plt.rcParams['figure.figsize'] = (15,7)
sns.scatterplot(x = df_week.index, y = df_week.Value,color = 'black')
plt.title('Foreign Exchane Rate(Weekly) - Euro to USD[Scatter Plot]')
plt.savefig('Foreign Exchane Rate(Weekly) - Euro to USD[Scatter Plot].png')
plt.show()


# _**Plotting the bar graph using seaborn to show the data spread in the resampled weekly data and saving the graph as PNG file**_

# In[18]:


sns.barplot(data = df_week, x = df_week.index, y = df_week.Value)
plt.title('Data Spread of Foreign Exchange Rate through Bar Plot')
plt.savefig('Data Spread of Foreign Exchange Rate through Bar Plot.png')
plt.show()


# _**Plotting the dist plot using Seaborn to show the data distribution in the resampled weekly dataset and saving the graph as PNG file**_

# In[19]:


sns.distplot(df_week)
plt.title('Distrubution of Data in Foreign Exchange Rate - Euro to USD')
plt.savefig('Distrubution of Data in Foreign Exchange Rate - Euro to USD.png')
plt.show()


# _**Plotting the histogram and KDE line graph to show the distribution of data in the resampled weekly dataset and saving the graph as PNG file**_

# In[20]:


fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = False, sharey = False, figsize = (10,5))
df_week.hist(ax = ax1)
df_week.plot(kind = 'kde', ax = ax2)
plt.title('Data Distribution of Foreign Exchange Rate')
plt.savefig('Data Distribution of Foreign Exchange Rate.png')
plt.show()


# _**Plotting the graph with "Seasonal Decompose" function to show the Data Description, Trend, Seasonal, Residuals and saving the graph as PNG file**_

# In[21]:


plt.rcParams['figure.figsize'] = (10,5)
decomposition = seasonal_decompose(df_week, period = 52, model = 'additive')
decomposition.plot()
plt.savefig('Trend, Seasonal, Residual Graph.png')
plt.show()


# _**Plotting the graphs with Auto-Correlation and Partial Auto-Correlation of the data from the resampled weekly dataset and saving the graphs as PNG file**_

# In[22]:


fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = False, sharey = False, figsize = (10,5))
ax1 = plot_acf(df_week, lags = 5, ax = ax1)
ax2 = plot_pacf(df_week, lags = 5, ax = ax2)
plt.subplots_adjust(hspace = 0.5)
plt.savefig('ACF & PACF.png')
plt.show()


# ## Data Transformation

# _**The `adf_check()` function performs an Augmented Dickey-Fuller test on a time series. The test is used to determine whether a time series is stationary or not. If the p-value of the test is less than or equal to 0.05, then there is strong evidence against the null hypothesis, and the series is considered to be stationary. Otherwise, the series is considered to be non-stationary. The function takes a time series as input and returns the results of the test. The results are printed to the console, along with a message indicating whether the series is stationary or not.**_

# In[23]:


def adf_check (time_series):
    result = adfuller (time_series)
    print ('Augmented Dickey Fuller Test :')
    labels = ['ADF Test Statistics', 'P Value', 'Number of Lags Used','Number of Observations']
    for value, label in zip(result, labels):
        print (label +' : '+ str (value))
    if result [1] <= 0.05:
        print ('Strong evidence against the null hypothesis, hence REJECT null hypothesis and the series is Stationary ')
    else:
        print ('Weak evidence against the null hypothesis, hence ACCEPT null hypothesis and the series is Not Stationary ')


# _**Performing the Augmented Dickey-Fuller test on the original data in the resampled weekly dataset to find whether the time series is stationary or not**_

# In[24]:


adf_check(df_week)


# _**Since the Time Series is Not Stationary, the resampled weekly dataset is transformed as a new DataFrame with First Difference to make it as a Stationary Series. Along with that showing the number of observations in the new DataFrame and displaying the first five records of the new DataFrame**_

# In[25]:


df1_week = df_week.diff().dropna()
print('Counts of the Weekly First Difference DataFrame  : ',df1_week.shape[0])
df1_week.head()


# _**Performing again the Augmented Dickey-Fuller test on the new transformed data from the resampled weekly dataset to find whether the time series is stationary or not**_

# In[26]:


adf_check(df1_week)


# _**Plotting the line graph to show the data trend in the transformed data from the resampled weekly dataset and saving the graph as PNG file**_

# In[27]:


df1_week.plot(figsize = (10,5))
plt.title('Foreign Exchane Rate(Weekly First Difference) - Euro to USD')
plt.savefig('Foreign Exchane Rate(Weekly First Difference) - Euro to USD.png')
plt.show()


# _**Plotting the graph with pandas plotting autocorrelation_plot to show the difference between the Stationary Data and Non-Stationary Data and saving it as PNG file**_

# In[28]:


fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = False, sharey = False, figsize = (10,5))
ax1 = autocorrelation_plot(df_week, ax = ax1)
ax1.set_title('Non-Stationary Data')
ax2 = autocorrelation_plot(df1_week, ax = ax2)
ax2.set_title('Stationary Data')
plt.subplots_adjust(hspace = 0.5)
plt.savefig('Autocorrelation_plot of Stationary & Non-Stationary.png')
plt.show()


# ## Model Fitting

# _**Getting the p value and q value for the model fitting using `auto_arima` function by passing through some needed parameters, the best model is evaluated by least Akaike Information Criterion[AIC]**_ 

# In[29]:


model = auto_arima(df_week, m = 52, d = 1, seasonal = False, max_order = 8, test = 'adf', trace = True)


# _**Defining the summary of the model fitted with `auto_arima` function, here getting various information such as Akaike Information Criterion[AIC], Bayesian Information Criterion[BIC}, Hannan-Quinn Information Criterion[HQIC], Log Likelihood etc. from which we can evaluate the model**_

# In[30]:


model.summary()


# _**Fitting the model in ARIMA model with the best value got from `auto_arima` model in the resampled weekly data and getting the summary of the fitted model**_

# In[31]:


model = ARIMA(df_week, order = (0,1,1))
result = model.fit()
result.summary()


# _**Plotting the Diagnostic plot for the fitted model to show the best fit of the model and saving it as PNG file**_

# In[32]:


result.plot_diagnostics(figsize = (15,5))
plt.subplots_adjust(hspace = 0.5)
plt.savefig('Diagnostic Plot of Best Model')
plt.show()


# _**Predicting the values using fitted model with whole resampled weekly data**_

# In[33]:


prediction = result.predict(typ = 'levels')


# ## Model Evaluation

# _**Evaluating the model with the following metrics such asPercentage of  R2 Score, Mean Squared Error, Root Mean Squared Error, Mean Absolute Error and Mean Absolute Percentage Error for the predicted value and whole data value**_

# In[34]:


print('Evaluation Results for Whole Data : \n')
print(' Percenatge of R2 Score : {0:.2f} %'.format(100*(r2_score(df_week['Value'],prediction))),'\n')
print(' Mean Squared Error : %.6f'%(mean_squared_error(df_week['Value'],prediction)),'\n')
print(' Root Mean Squared Error : ',sqrt(mean_squared_error(df_week['Value'],prediction)),'\n')
print(' Mean Absolute Error : ',mean_absolute_error(df_week['Value'],prediction),'\n')
print(' Mean Absolute Percentage Error : {0:.2f} %'.format(100*mean_absolute_percentage_error(df_week['Value'],prediction)),'\n')


# ## Result

# _**Concating the Resampled weekly dataset, Weekly First Difference DataFrame and the prediction value and naming the columns as `Foreign Exchange Rate(Weekly)`, `Weekly First Difference`, `Predicted Exchange Rate` and exporting the DataFrame to [Comma Seperated Value]csv file. Also displaying the first five records of the exported DataFrame**_

# In[35]:


df_final = pd.concat([df_week, df1_week, prediction],axis = 1)
df_final.columns = ['Foreign Exchange Rate(Weekly)', 'Weekly First Difference', 'Predicted Exchange Rate']
df_final.to_csv('Foreign Exchange Rate with Prediction (Euro to USD).csv')
df_final.head()


# ## Model Testing

# _**Splitting the Resampled Weekly Dataset into training data and test data. And displaying the number of observations in both training data nad test data**_

# In[36]:


size = int(len(df_week)*0.80)
train, test = df_week[0:size]['Value'], df_week[size:(len(df_week))]['Value']
print('Counts of the Train Data : ',train.shape[0])
print('Counts of the Test Data : ',test.shape[0])


# _**Creating the list of train dataset values in train_values and empty predictions list which will be appended after the prediction. Then fitting the model with ARIMA model with the best value got from auto_arima model in the train_values data and predicting with test data value and appending it to the predictions list and printing the comparison between predicted value and actual value**_

# In[37]:


train_values = [x for x in train]
predictions = []
print('Printing Predicted vs Expected Values...')
print('\n')
for t in range(len(test)):
    model = ARIMA(train_values, order=(0,1,1))
    model_fit = model.fit()
    output = model_fit.forecast()
    pred_out = output[0]
    predictions.append(float(pred_out))
    test_in = test[t]
    train_values.append(test_in)
    print('predicted=%f, Actual=%f' % (pred_out, test_in))


# _**Evaluating the model with the following metrics such asPercentage of R2 Score, Mean Squared Error, Root Mean Squared Error, Mean Absolute Error and Mean Absolute Percentage Error for the predicted value and test data**_

# In[38]:


print('Evaluation Results for Test Data : \n')
print(' Percenatge of R2 Score : {0:.2f} %'.format(100*(r2_score(test,predictions))),'\n')
print(' Mean Squared Error : %.6f'%(mean_squared_error(test,predictions)),'\n')
print(' Root Mean Squared Error : ',sqrt(mean_squared_error(test,predictions)),'\n')
print(' Mean Absolute Error : ',mean_absolute_error(test,predictions),'\n')
print(' Mean Absolute Percentage Error : {0:.2f} %'.format(100*mean_absolute_percentage_error(test,predictions)),'\n')


# _**Making the predicted value as Series and index as same as Test data index since the prediction is based on Test Data**_

# In[39]:


predictions_df = pd.Series(predictions, index = test.index)


# _**Plotting the graph with predicted data and the resampled weekly data and saving it as PNG file**_

# In[40]:


fig, ax = plt.subplots()
ax.set(title='Foreign Exchange Rate Prediction- Euro to USD', xlabel='Date', ylabel='Foreign Exchange Rate')
ax.plot(df_week, 'o', label='Actual')
ax.plot(predictions_df, 'r', label='Forecast')
legend = ax.legend(loc='upper right')
legend.get_frame().set_facecolor('w')
plt.savefig('Foreign Exchange Rate Prediction- Euro to USD.png')


# _**Creating the pickle file with the best model that gives high evaluation score for the test data**_

# In[41]:


pickle.dump(model_fit,open('Best Model.pkl','wb'))

