#!/usr/bin/env python
# coding: utf-8

# # IRIS  FLOWER  CLASSIFICATION
# 

# In[14]:


import pandas as pd
value= pd.read_csv("IRIS.csv")


# In[15]:


a = value.drop('species', axis=1)
b = value['species']


# In[16]:


from sklearn.model_selection import train_test_split

a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.2, random_state=42)


# In[17]:


from sklearn.svm import SVC

MDL = SVC(kernel='linear', C=1.0, random_state=42)
MDL.fit(a_train, b_train)


# In[18]:


from sklearn.metrics import accuracy_score

b_prediction = MDL.predict(a_test)
accur = accuracy_score(b_test, b_prediction)
print(f'Accuracy: {accur:.2f}')


# In[22]:


new_data = pd.DataFrame({'sepal_length': [3.4], 'sepal_width': [3.9], 'petal_length': [4.8], 'petal_width': [1.5]})
prediction = MDL.predict(new_data)
print(f'Prediction: {prediction[0]}')


# In[3]:


import os


# In[7]:


os.getcwd()


# In[6]:


os.chdir("C:\\Users\\ronit\\Downloads\\IRIS.csv")

