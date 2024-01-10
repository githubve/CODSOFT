#!/usr/bin/env python
# coding: utf-8

# Import Libraries

# In[21]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm


# Import data

# In[31]:


data = pd.read_csv("C:/Users/ADMIN/Downloads/archive (5)/spam.csv")
data


# In[23]:


data.info()


# Data Cleaning

# In[24]:


data=data[['v1','v2']].copy()
data.rename(columns={'v1':'class','v2':'text'},inplace=True)
data['target']=data['class'].map({'ham':0 ,'spam':1})
new_data=data[['target','text']]

# dublicate values
new_data.duplicated().sum()


# In[25]:


new_data.drop_duplicates(inplace=True)


# In[26]:


new_data.duplicated().sum()


# In[27]:


new_data


# Data Visualization

# In[28]:


new_data['target'].value_counts().plot(kind='bar')
plt.xlabel('target')
plt.title('ham:0' and 'spam:1')
plt.show()


# Text Data Preprocessing

# In[32]:


data.drop_duplicates(inplace=True)
data['label'] = data['v1'].map({'ham': 'ham', 'spam': 'spam'})
X = data['v2']
y = data['label']


# Split the data into two sets : Training Set and Testing Set

# In[33]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Tf-Idf Vecctorizer

# In[34]:


#Create a Tf-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

#Fit the vectorizer to the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)


# In[35]:


#Initialize a Naive Bayes Classifier 
classifier = MultinomialNB()

#Train the classifier
classifier.fit(X_train_tfidf, y_train)


# In[36]:


#Transform the test data 
X_test_tfidf = tfidf_vectorizer.transform(X_test)

#Make Predictions
y_pred = classifier.predict(X_test_tfidf)


# In[37]:


#Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

#Display classification report with labels 'ham' and 'spam'
report = classification_report(y_test, y_pred, target_names=['Legitimate SMS', 'Spam SMS'])


# In[38]:


#Create a progress Bar 
progress_bar = tqdm(total=100, position=0, leave=True)

#Simulate progress updates
for i in range(10, 101, 10):
    progress_bar.update(10)
    progress_bar.set_description(f'Progress: {i}%')


# In[39]:


#Close Progress bar
progress_bar.close()

#Display the results on the interface where the code was initiated from
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)


# Thanks for Watching

# In[ ]:




