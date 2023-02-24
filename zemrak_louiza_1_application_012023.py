#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 18:32:24 2023

@author: zemrak
"""

# Start importing relevant librairies
# Import libraries
import os
import pandas as pd
import numpy as np
from time import time
from random import randint
from sklearn.metrics.pairwise import cosine_similarity

from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

from tqdm import tqdm

from implicit.lmf import LogisticMatrixFactorization
from implicit.evaluation import precision_at_k, mean_average_precision_at_k, ndcg_at_k, AUC_at_k
import pickle
import flask
from flask import jsonify


clicks = pd.read_csv("https://github.com/lz60/recommandation/releases/download/clicks/clicks.csv")
MODEL_PATH = "./recommender.model"
if not os.path.exists(MODEL_PATH):
    os.system("wget https://github.com/lz60/recommandation/releases/download/model/recommender.model")

def compute_interaction_matrix(clicks):
    # Create interaction DF (count of interactions between users and articles)
    interactions = clicks.groupby(['user_id', 'article_id']).size().reset_index(name='count')
    # print('Interactions DF shape: ', interactions.shape)

    # csr = compressed sparse row (good format for math operations with row slicing)
    # Create sparse matrix of shape (number_items, number_user)
    csr_item_user = csr_matrix((interactions['count'].astype(float),
                                (interactions['article_id'],
                                 interactions['user_id'])))
    # print('CSR Shape (number_items, number_user): ', csr_item_user.shape)

    # Create sparse matrix of shape (number_user, number_items)
    csr_user_item = csr_matrix((interactions['count'].astype(float),
                                (interactions['user_id'],
                                 interactions['article_id'])))
    # print('CSR Shape (number_user, number_items): ', csr_user_item.shape)

    return csr_item_user, csr_user_item


def get_cf_reco(clicks, userID, csr_item_user, csr_user_item, model_path=None, n_reco=5, train=True):
    start = time()
    # Train the model on sparse matrix of shape (number_items, number_user)

    if train or model_path is None:
        model = LogisticMatrixFactorization(factors=128, random_state=42)
        print("[INFO] : Start training model")
        model.fit(csr_user_item)

        # Save model to disk
        with open('recommender.model', 'wb') as filehandle:
            pickle.dump(model, filehandle)
    else:
        with open(MODEL_PATH, 'rb') as filehandle:
            model = pickle.load(filehandle)

    # Recommend N best items from sparse matrix of shape (number_user, number_items)
    # Implicit built-in method
    # N (int) : number of results to return
    # filter_already_liked_items (bool) : if true, don't return items present in
    # the training set that were rated/viewd by the specified user
    recommendations_list = []
    recommendations = model.recommend(userID, csr_user_item[userID], N=n_reco, filter_already_liked_items=True)

    print(f'[INFO] : Completed in {round(time() - start, 2)}s')
    print(f'[INFO] : Recopendations for user {userID}: {recommendations[0].tolist()}')

    return recommendations[0].tolist()


csr_item_user, csr_user_item = compute_interaction_matrix(clicks)

app = flask.Flask(__name__)

# This is the route to the API
@app.route("/")
def home():
    return "Welcome on the recommendation API ! "

@app.route("/get_recommendation/<id>", methods=["POST", "GET"])
def get_recommendation(id):

    recommendations = get_cf_reco(clicks, int(id), csr_item_user, csr_user_item, model_path=MODEL_PATH, n_reco=5, train=False)
    data = {
            "user" : id,
            "recommendations" : recommendations,
        }
    return jsonify(data)