#!/usr/bin/env python
# coding: utf-8

# In[1]:


import difflib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


# In[2]:


df2 = pd.read_csv(r'C:\Users\91900\Desktop\recommendation-system-master\model\tmdb.csv')
df2.head()


# In[3]:


df2.info()


# In[4]:


df2.dropna(axis=0)


# In[5]:


df2.isnull().sum()


# In[6]:


df2=df2.drop(['tagline','id','homepage',],axis=1)
df2.isnull().sum()


# In[7]:


count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title'])

def get_recommendations(title):
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    tit = df2['title'].iloc[movie_indices]
    dat = df2['release_date'].iloc[movie_indices]
    return_df = pd.DataFrame(columns=['Title','Year'])
    return_df['Title'] = tit
    return_df['Year'] = dat
    return return_df


# In[14]:


pickle.dump(cosine_sim2,open('model.pkl','wb'))


# In[15]:


model=pickle.load(open('model.pkl','rb'))


# In[ ]:




