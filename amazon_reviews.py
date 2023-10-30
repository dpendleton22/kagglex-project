import streamlit as st
import pandas as pd
import ast
import torch
import altair as alt
import numpy as np
import pickle
from PIL import Image
import requests
from io import BytesIO
import urllib.request
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.util import semantic_search
from transformers import pipeline

@st.cache_data
def _load_data():
    return pd.read_csv('data/dewalt_no_dupes_final.csv')

@st.cache_data
def _load_embeds():
    return torch.load('mpnet_pruned_nodupes_embeds.pt')

@st.cache_resource
def _load_summarizer():
    return pipeline("summarization", model="lidiya/bart-large-xsum-samsum")

def _get_question_results(question):
    query_embedding = embedder.encode(question, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, embeds)[0]
    top_results = torch.topk(cos_scores, k=10)

    results_idx = [x for x in top_results[1]]

    results_df = df.iloc[results_idx]

    return results_df

def _post_reviews(reviews_df):
    for index, row in reviews_df.iterrows():
        st.write(row['fullReviewText'])

def _plot_product(product_name):
    prod_df = df.loc[df['title'] == product_name]
    prod_df['review_year'] = pd.to_datetime(prod_df['reviewTime_dt']).dt.year
    prod_reviews = prod_df.groupby(['review_year', 'overall']).count()
    if prod_df['price_adj'].iloc[0] == -99 or len(ast.literal_eval(prod_df['imageURLHighRes'].iloc[0])) == 0:
        return

    prod_reviews.reset_index(inplace=True)
    print (prod_reviews)
    with st.container():
        st.header(f"Product: {prod_df['title'].iloc[0]}")
        st.subheader(f"Category: {prod_df['third_category'].iloc[0]}")
        st.write(f"Price: ${prod_df['price_adj'].iloc[0]}")
        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            image_res = ast.literal_eval(prod_df['imageURLHighRes'].iloc[0])
            response = requests.get(image_res[0])
            image = Image.open(BytesIO(response.content))
            st.image(image, width = 200)
        with col2:
            chart = alt.Chart(prod_reviews).mark_line().encode(
                x='review_year',
                y='asin', color='overall').encode(alt.X('review_year', title='Year'), 
                                                  alt.Y("asin", title='Num. Reviews'))
            st.altair_chart(chart, use_container_width=True)

df = _load_data()
embeds = _load_embeds()
corpus = df['fullReviewText'].to_list()
summarizer = _load_summarizer()


embedder = SentenceTransformer('all-mpnet-base-v2') #all-distilroberta-v1 

st.title('Know Your Customers')

with st.sidebar:
    st.header('Product Reviews Summary')
    worst_products = df[['title', 'third_category', 'imageURLHighRes', 'overall']].loc[df['overall'] == 1].groupby(['title', 'third_category', 'imageURLHighRes']).count()
    best_products = df[['title', 'third_category', 'imageURLHighRes', 'overall']].loc[df['overall'] == 5].groupby(['title', 'third_category', 'imageURLHighRes']).count()

    st.header('Best Reviewed Products')
    sorted_best = best_products.sort_values('overall', ascending=False).head(4)
    print (sorted_best)
    sorted_best.reset_index(inplace=True)

    for x in range(0, len(sorted_best)):
        with st.container():
            col1, col2 = st.columns([0.3, 0.7])
            with col1:
                image_res = ast.literal_eval(sorted_best['imageURLHighRes'].iloc[x])
                response = requests.get(image_res[0])
                best_image = Image.open(BytesIO(response.content))
                st.image(best_image, width=70)
            with col2:
                st.caption(sorted_best['title'].iloc[x])
                st.text(f"5 Star Ratings: {sorted_best['overall'].iloc[x]}")
        st.divider()

    st.subheader('Lowest Reviewed Products')
    sorted_worst = worst_products.sort_values('overall', ascending=False).head(4)
    sorted_worst.reset_index(inplace=True)

    for y in range(0, len(sorted_worst)):
         with st.container():
            col1, col2 = st.columns([0.3, 0.7])
            with col1:
                image_res = ast.literal_eval(sorted_worst['imageURLHighRes'].iloc[y])
                if len(image_res) == 0:
                    continue
                response = requests.get(image_res[0])
                worst_image = Image.open(BytesIO(response.content))
                st.image(worst_image, width=70)
            with col2:
                st.caption(sorted_worst['title'].iloc[x])
                st.text(f"1 Star Ratings: {sorted_worst['overall'].iloc[y]}")
         st.divider()


    # for i in range(0, len(sorted_worst)):
    #     product_df = df.loc[df['title'] == sorted_worst['title'][i]][0]
    #     image = Image.open(product['imageURLHighRes'].loc[i])
    #     st.image(image, caption=sorted_worst['title'].loc[i])

user_question = st.text_input(label='Type questions here', placeholder="How do users feel about the quality of our batteries??")
options = st.multiselect(
    'Select a specific category of products for your question',
    df['secondary_category'].unique(), placeholder='Leave empty if you want to include all categories of products')
if user_question:
    output_df = _get_question_results(user_question)

    all_reviews = ''.join(output_df['fullReviewText'].to_list())
    all_summary = summarizer(all_reviews)

    print (all_summary)
    st.info(all_summary[0]['summary_text'])

    emo_count = output_df.groupby('primary_emo').count()
    st.bar_chart(emo_count, y='asin')

    _post_reviews(output_df)

    st.text(output_df['title'].unique())

    for i in output_df['title'].unique():
        _plot_product(i)
