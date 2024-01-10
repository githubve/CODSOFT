#!/usr/bin/env python
# coding: utf-8

# Importing Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[ ]:


credit_card_data = pd.read_csv('C:/Users/ADMIN/Downloads/archive (3)/fraudTest.csv', parse_dates=['trans_date_trans_time',])
credit_card_data


# In[ ]:


# column of dataset 
credit_card_data.columns


# Adding Hour Features

# In[ ]:


credit_card_data['hour'] = credit_card_data.trans_date_trans_time.dt.hour


# In[ ]:


credit_card_data['hourEnc'] = 0
credit_card_data.loc[credit_card_data.hour < 5,'hourEnc'] = 1
credit_card_data.loc[credit_card_data.hour > 21,'hourEnc'] = 1


# In[ ]:


# Assuming 'trans_date_trans_time' is a datetime column
credit_card_data['trans_date_trans_time'] = pd.to_datetime(credit_card_data['trans_date_trans_time'])

# Sorting the DataFrame based on the transaction time
credit_card_data.sort_values(by='trans_date_trans_time', inplace=True)

# Adding frequencies for last 1, 7, and 30 days
credit_card_data['transactions_last_1d'] = credit_card_data['trans_date_trans_time'].rolling(window=1).count()
credit_card_data['transactions_last_7d'] = credit_card_data['trans_date_trans_time'].rolling(window=7).count()
credit_card_data['transactions_last_30d'] = credit_card_data['trans_date_trans_time'].rolling(window=30).count()

# Filling NaN values with 0 for the initial rows with no history
credit_card_data['transactions_last_1d'].fillna(0, inplace=True)
credit_card_data['transactions_last_7d'].fillna(0, inplace=True)
credit_card_data['transactions_last_30d'].fillna(0, inplace=True)


# In[ ]:


# Assuming credit_card_data is your DataFrame
credit_card_data['trans_date_trans_time'] = pd.to_datetime(credit_card_data['trans_date_trans_time'])

def last1DayTransactionCount(group):
    return group['trans_date_trans_time'].rolling(window=1).count()

def last7DaysTransactionCount(group):
    return group['trans_date_trans_time'].rolling(window=7).count()

def last30DaysTransactionCount(group):
    return group['trans_date_trans_time'].rolling(window=30).count()

# Group by 'cc_num' and apply the custom functions
df1 = credit_card_data.groupby('cc_num').apply(last1DayTransactionCount)
df1 = credit_card_data.groupby('cc_num').apply(last7DaysTransactionCount)
df1 = credit_card_data.groupby('cc_num').apply(last30DaysTransactionCount)

# Resetting the index to obtain a DataFrame
df1 = df1.reset_index(level=0, drop=True)


# In[ ]:


# Assuming credit_card_data is your DataFrame
credit_card_data['trans_date_trans_time'] = pd.to_datetime(credit_card_data['trans_date_trans_time'])

def last1DayTransactionCount(group):
    return group.resample('1D', on='trans_date_trans_time').size()

def last7DaysTransactionCount(group):
    return group.resample('7D', on='trans_date_trans_time').size()

def last30DaysTransactionCount(group):
    return group.resample('30D', on='trans_date_trans_time').size()

def timeSinceLastTransaction(group):
    return group['trans_date_trans_time'].diff().dt.total_seconds()

# Group by 'cc_num' and apply the custom functions
df1 = credit_card_data.groupby('cc_num').apply(last1DayTransactionCount)
df1 = credit_card_data.groupby('cc_num').apply(last7DaysTransactionCount)
df1 = credit_card_data.groupby('cc_num').apply(last30DaysTransactionCount)
df1['time_diff'] = credit_card_data.groupby('cc_num').apply(timeSinceLastTransaction)

# Resetting the index to obtain a DataFrame
df1 = df1.reset_index(level=0, drop=True)


# Displaying the correlation between the features

# In[ ]:


# Drop non-numeric columns
numeric_data = credit_card_data.select_dtypes(include=['number'])

# Calculate correlation matrix
correlation_matrix = numeric_data.corr()

# Plotting a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# In[ ]:


# Assuming credit_card_data is your DataFrame
numeric_columns = credit_card_data.select_dtypes(include='number')
fraud_correlation = numeric_columns.corr()['is_fraud'].abs().sort_values(ascending=False)

print(fraud_correlation)


# In[ ]:


# Load train and test datasets
train_data = pd.read_csv('C:/Users/ADMIN/Downloads/archive (3)/fraudTest.csv')
test_data = pd.read_csv('C:/Users/ADMIN/Downloads/archive (3)/fraudTrain.csv')

# Drop non-numeric and non-binary columns for simplicity
numeric_columns_train = train_data.select_dtypes(include='number')
numeric_columns_test = test_data.select_dtypes(include='number')

X_train = numeric_columns_train.drop(['is_fraud'], axis=1)
y_train = train_data['is_fraud']

X_test = numeric_columns_test.drop(['is_fraud'], axis=1)
y_test = test_data['is_fraud']

# Preprocess the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build and train the Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{classification_rep}")


# Thanks for Watching

# In[ ]:




