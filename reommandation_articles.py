#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 21:10:28 2023

@author: zemrak
"""


import streamlit as st
import pandas as pd

import requests
import numpy as np
import  json


st.write("""
# application de recommandation d'articles'
""")
#id_users = pd.read_csv('/app/recommandation/id.csv')
#liste_id=id_users.user_id.tolist()
	
id = st.slider("Choisir un id  UTILISATEUR: ", 0,900)
#id=0
#id = st.selectbox(
#   "choisisez un id  dans la liste  ci-dessous:  ",
#     ( liste_id))
	



  
# Set the URL of the Azure function
azure_url = "https://sys-recommandation.azurewebsites.net/api/recommandation_articles"
# for example : azure_url = "https://netflix_recommendations.azurewebsites.net/api/get_recommandations/"

# Set the parameters of the request
request_params = {"user_id":id}

# Send the request to the Azure function
r = requests.post(azure_url, params=request_params)
    
    

# Grab the recommendations as a Python dictionary
recommendations_dict = json.loads(r.content.decode())
st.write("les articles recommandés pour cette utilisateur sont :", recommendations_dict )
print(recommendations_dict)
