#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import streamlit as st
import pickle
from sklearn import datasets
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.linear_model import LogisticRegression


# In[4]:


st.write("""
# This is Blood Donation Prediction App
This app predicts the **Blood Donation** for Future Expectancy!
""")


# In[5]:


st.sidebar.header('User Input Parameters')
def user_input_features():
    Recency = st.sidebar.slider('Recency(months)', 0, 30, 21)
    Frequency = st.sidebar.slider('Frequency(times)',0, 60, 1)
    Monetary  = st.sidebar.slider('Monetary (c.c. blood)', 100, 15000, 250)
    Time = st.sidebar.slider('Time (months)', 1, 100, 21)
   # Monetary=np.log( Monetary)
    #Time=np.log( Time)
        
#features["Time (months)"]=np.log(features["Time (months)"])
    data = {'Recency (months)': Recency,
            'Frequency (times)': Frequency,
            'Monetary (c.c. blood)': Monetary,
            'Time (months)': Time}
    feature = pd.DataFrame(data, index=[0])
    return feature

df = user_input_features()


# In[6]:


st.subheader('User Input parameters')
st.write(df)


# In[7]:
df["Monetary (c.c. blood)"]=np.log(df["Monetary (c.c. blood)"])
df["Time (months)"]=np.log(df["Time (months)"])




# Reads in saved classification model
load_clf = pickle.load(open('blood_clf.pkl', 'rb'))



predictions=load_clf.predict(df)


# In[10]:


st.subheader('Prediction')
st.write(predictions[0])

