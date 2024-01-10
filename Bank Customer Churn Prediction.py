#!/usr/bin/env python
# coding: utf-8

# Importing Libraries




import numpy as np # linear algebra
import pandas as pd 

# Input data files are available in the read-only "../input/" directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))





import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Loading Dataset

# In[3]:


data = pd.read_csv("C:/Users/ADMIN/Downloads/archive (4)/Churn_Modelling.csv")
data


# Data Analysis

# In[4]:


print(data.info())


# In[5]:


print(data.describe())


# In[6]:


plt.figure(figsize=(15,5))
sns.countplot(data=data,x='Exited')


# In[7]:


data['Exited'].value_counts().to_frame()


# In[8]:


from sklearn.utils import resample


# In[9]:


churn_majority=data[data['Exited']==0]
churn_minority=data[data['Exited']==1]


# In[10]:


churn_majority_downsample=resample(churn_majority,n_samples=2037,replace=False,random_state=42)


# In[11]:


churn_df=pd.concat([churn_majority_downsample,churn_minority])


# In[12]:


churn_df['Exited'].value_counts().to_frame()


# In[13]:


plt.figure(figsize=(15,5))
sns.countplot(data=churn_df,x='Exited')


# In[14]:


churn_df.columns


# In[15]:


churn_df.drop(['RowNumber', 'CustomerId', 'Surname','Geography','Gender'], axis=1, inplace=True)


# In[16]:


churn_df.corr()


# In[17]:


plt.figure(figsize=(15,5))
sns.heatmap(churn_df.corr(),annot=True)


# In[18]:


df_corr_exit=churn_df.corr()['Exited'].to_frame()
plt.figure(figsize=(15,5))
sns.barplot(data=df_corr_exit,x=df_corr_exit.index,y='Exited')


# Data Preprocessing

# In[19]:


# Drop irrelevant columns (e.g., RowNumber, CustomerId, Surname)
df = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Handle categorical variables (Geography and Gender) using one-hot encoding
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

# Split the data into features (X) and the target variable (y)
X = df.drop('Exited', axis=1)
y = df['Exited']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (optional but often improves model performance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Model Selection and Training

# In[20]:


# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)


# Model Evaluation

# In[21]:


# Make predictions
lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
gb_pred = gb_model.predict(X_test)

# Evaluate model performance
print("Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print("Classification Report:\n", classification_report(y_test, lr_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))

print("\nRandom Forest:")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Classification Report:\n", classification_report(y_test, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))

print("\nGradient Boosting:")
print("Accuracy:", accuracy_score(y_test, gb_pred))
print("Classification Report:\n", classification_report(y_test, gb_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, gb_pred))


# In[22]:


lr_model.score(X_train,y_train)


# Model Interpretation

# In[23]:


# Logistic Regression Coefficients
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr_model.coef_[0]
})
print(coefficients.sort_values(by='Coefficient', ascending=False))


# In[24]:


# Set the style for seaborn
sns.set(style="whitegrid")

# Distribution of the target variable (Exited)
plt.figure(figsize=(6, 4))
sns.countplot(x='Exited', data=df, palette='Set2')
plt.title('Distribution of Exited')
plt.show()


# In[25]:


# Correlation matrix
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# In[26]:


# Pairplot for selected features
selected_features = ['CreditScore', 'Age', 'Balance', 'NumOfProducts', 'EstimatedSalary']
sns.pairplot(df, vars=selected_features, hue='Exited', palette='husl', markers=['o', 's'])
plt.suptitle('Pairplot of Selected Features', y=1.02)
plt.show()


# In[27]:


# Boxplot for Age and CreditScore
plt.figure(figsize=(12, 6))
sns.boxplot(x='Exited', y='Age', data=df, palette='Set2')
plt.title('Boxplot of Age by Churn')
plt.show()


# In[28]:


plt.figure(figsize=(12, 6))
sns.boxplot(x='Exited', y='CreditScore', data=df, palette='Set2')
plt.title('Boxplot of CreditScore by Churn')
plt.show()


# Thanks for Watching

# In[ ]:




