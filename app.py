#!/usr/bin/env python
# coding: utf-8

# In[2]:


import flask
import difflib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

df2 = pd.read_csv(r'C:\Users\91900\Desktop\recommendation-system-master\model\tmdb.csv')
df2.head()


# In[3]:


df2.info()


# In[4]:


df2.dropna(axis=0)


# In[6]:


df2.isnull().sum()


# The id column, which is unique for each movie will not contribute to the recommendations
# The tagline column should be eliminated because most of the movies have an overview and thus the tagline would result in more of a similar context
# Homepage has a lot of empty values, we have no other option but to drop it.
# Since we have one move which is unreleased, we can drop that particular row, since the movie is unreleased. 
# Now we have our final dataset which is ready for some machine learning modeling.

# In[7]:


df2=df2.drop(['tagline','id','homepage',],axis=1)
df2.isnull().sum()


# In[ ]:


app = flask.Flask(__name__, template_folder='templates')
model=pickle.load(open('model.pkl','rb'))
all_titles = [df2['title'][i] for i in range(len(df2['title']))]

#This allows flask to receive information from the form and display appropriate info on the results page
@app.route('/', methods=['GET', 'POST'])

def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))
            
    if flask.request.method == 'POST':
        m_name = flask.request.form['movie_name'] #takes all the values from the textfield and stores under m_name
        m_name = m_name.title() #converted into an array
        if m_name not in all_titles:
            return(flask.render_template('negative.html',name=m_name))
        else:
            result_final = get_recommendations(m_name)
            names = []
            dates = []
            for i in range(len(result_final)):
                names.append(result_final.iloc[i][0])
                dates.append(result_final.iloc[i][1])

            return flask.render_template('positive.html',movie_names=names,movie_date=dates,search_name=m_name)

if __name__ == '__main__':
    app.run()


# In[ ]:




