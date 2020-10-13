#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd


# In[8]:


df=pd.read_csv(r"E:\Krish naik\NLP\Stock-Sentiment-Analysis-master\Data.csv",encoding='latin1')


# In[9]:


df.head()


# # Train-Test Split

# In[10]:



train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']


# In[12]:


# Removing punctuations
data=train.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)

# Renaming column names for ease of access
list1= [i for i in range(25)]
new_Index=[str(i) for i in list1]
data.columns= new_Index
data.head(5)


# In[13]:


# Convertng headlines to lower case
for index in new_Index:
    data[index]=data[index].str.lower()
data.head(1)


# In[14]:


# combining all text into paragraph
headlines = []
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))


# In[15]:



headlines


# In[16]:



' '.join(str(x) for x in data.iloc[1,0:25])


# In[17]:



headlines[0]


# In[20]:



from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


# In[21]:


## implement BAG OF WORDS
countvector=CountVectorizer(ngram_range=(2,2))
traindataset=countvector.fit_transform(headlines)


# In[ ]:


# Training the Word2Vec model
model = Word2Vec(sentences, min_count=1)


# In[22]:


# implement RandomForest Classifier
randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(traindataset,train['Label'])


# In[ ]:


countvector=CountVectorizer()


# In[23]:


## Predict for the Test Dataset
test_transform= []
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = countvector.transform(test_transform)
predictions = randomclassifier.predict(test_dataset)


# In[24]:



## Import library to check accuracy
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[25]:


matrix=confusion_matrix(test['Label'],predictions)
print(matrix)
score=accuracy_score(test['Label'],predictions)
print(score)
report=classification_report(test['Label'],predictions)
print(report)


# In[ ]:




