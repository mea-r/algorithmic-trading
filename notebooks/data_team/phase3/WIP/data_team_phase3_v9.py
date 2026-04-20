#!/usr/bin/env python
# coding: utf-8

# ### **Import libraries**: 
# - **Hugging Face** is a platform and library for utilizing machine learning models.
# - **InferenceClient** is a library available on Hugging Face and is primarily used to perform inference – the process of reaching conclusions given a set of inputs.
# - **Transformers** is a Python library created by Hugging Face that allows one to download, manipulate, and run thousands of pretrained, open-source AI models.
# - **Pipeline** is a package within Transformers that simplifies the process of working with advanced AI models.
# - **UUID** is a library that provides a way to generate universally unique identifiers. In this script, it is used to create identifiers for headlines.

# In[1]:


import requests
from huggingface_hub import InferenceClient
from transformers import pipeline
import uuid
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from dateutil.rrule import rrule, DAILY
import random
import sys
import fastparquet
import os
from IPython.display import clear_output
import time


# ### **Create a data frame with ticker, headlineID, headline, article publication date, and article source columns.** ###

# In[2]:


def package(ticker_lst_param):
    df_main = pd.DataFrame(columns = ["Ticker", "HeadlineID", "Headline", "Timestamp", "Source"])
    for element in ticker_lst_param:
        tickers = [element["Ticker"] for i in range(len(element["Headline"]))]
        headline_IDs = [str(uuid.uuid4())[:8] for i in range(len(element["Headline"]))]
        data = {"Ticker" : tickers, "HeadlineID" : headline_IDs, "Headline" : element["Headline"], 
                "Timestamp" : element["Timestamp"], "Source" : element["Source"]}
        df = pd.DataFrame(data = data)
        df_main = pd.concat([df_main, df])
    return df_main


# ### **Use the Alpha Vantage API to pull news associated with each ticker.**
# 
# ### **To modify the tickers used, make changes to tickers.txt**

# In[3]:


def get_data(txt_file_containing_tickers, collection_start_date, collection_end_date, article_limit_per_day):
    with open(txt_file_containing_tickers, "r") as f:
        tickers = f.readline()
        tickers = tickers[1:-1]
        tickers = tickers.replace('"', "")
        tickers = tickers.split(",")
        tickers = [element.strip() for element in tickers]
    
    with open("api_keys/alpha_vantage.txt", "r") as f:
        alpha_vantage_api_key = f.readline().strip()

    limit = 50
    headline_data = []
    
    days = collection_end_date - collection_start_date + timedelta(days=1)

    total = len(tickers) * int(days.days)
    counter = 1
    
    for ticker in tickers:
        for day in rrule(DAILY, dtstart=collection_start_date, until=collection_end_date):

            #print(day)
            
            time_from, time_to = day, day + timedelta(days = 1)
            time_from, time_to = time_from.strftime("%Y%m%dT%H%M"), time_to.strftime("%Y%m%dT%H%M")
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&time_from={time_from}&time_to={time_to}&limit={limit}&apikey={alpha_vantage_api_key}"
            response = requests.get(url)
            data = response.json()

            try:
                news = data["feed"]
            except:
                print(data)
            
            headlines = [article["title"] for article in news]
            
            timestamps = [article["time_published"] for article in news]
            timestamps = [pd.to_datetime(timestamp, format = "mixed") for timestamp in timestamps]
            sources = [article["source"] for article in news]

            headlines_no_dupe, timestamps_no_dupe, sources_no_dupe = [], [], []


            for headline, timestamp, source in zip(headlines, timestamps, sources):
                if headline not in headlines_no_dupe:
                    headlines_no_dupe.append(headline)
                    timestamps_no_dupe.append(timestamp)
                    sources_no_dupe.append(source)


            headlines_no_dupe_limited, timestamps_no_dupe_limited, sources_no_dupe_limited = [], [], []

            if len(headlines_no_dupe) > 25:
                indexes_to_remove = random.sample(range(0, len(headlines_no_dupe)), len(headlines_no_dupe) - article_limit_per_day)
            else:
                indexes_to_remove = []
            for headline, timestamp, source in zip(headlines_no_dupe, timestamps_no_dupe, sources_no_dupe):
                if headlines_no_dupe.index(headline) not in indexes_to_remove:
                    headlines_no_dupe_limited.append(headline)
                    timestamps_no_dupe_limited.append(timestamp)
                    sources_no_dupe_limited.append(source)
            
            headline_datum = {"Ticker" : ticker, "Headline" : headlines_no_dupe_limited, 
                              "Timestamp" : timestamps_no_dupe_limited, "Source" : sources_no_dupe_limited}
            
            headline_data.append(headline_datum)

            with open("news_status.txt", "w") as f:
                f.write(str(round((counter/total)*100, 2)) + "% complete." + "\n")

            #print(str(round((counter/total)*100, 2)) + "% complete.", counter, " API calls.")
            
            counter +=1
            time.sleep(0.55)

    headline_data_packaged = package(headline_data)

    return headline_data_packaged


# ### **Utilize FinBERT - a pre-trained NLP model specialized for financial sentiment analysis – adapted by ProsusAI to analyze the probabilities (positive/negative/neutral) associated with each headline.**

# In[4]:


def add_probabilities(ticker_news_data_df_param, api_key_param):
    pipe = pipeline("text-classification", model="ProsusAI/finbert")
    client = InferenceClient(provider="hf-inference", api_key=api_key_param)
    ticker_news_data_df_param[["pos", "neu", "neg"]] = 0.0, 0.0, 0.0
    ticker_news_data_df_param["Timestamp"] = pd.to_datetime(ticker_news_data_df_param["Timestamp"])
    rows_num, row_counter = len(ticker_news_data_df_param), 1

    for index, row in ticker_news_data_df_param.iterrows():
        headline = row["Headline"]
        result = client.text_classification(headline, model="ProsusAI/finbert")
        for probability in result:
            row = row.copy()
            match probability["label"]:
                case "positive":
                    ticker_news_data_df_param.loc[index, "pos"] = probability.score
                case "neutral":
                    ticker_news_data_df_param.loc[index, "neu"] = probability.score
                case "negative":
                    ticker_news_data_df_param.loc[index, "neg"] = probability.score
        
        clear_output()
        
        with open("rating_status.txt", "w") as f:
            f.write(str(round((row_counter/rows_num)*100, 2)) + "% complete." + "\n")
        
        #print(str(round((row_counter/rows_num)*100, 2)) + "% complete.")

        row_counter +=1
        time.sleep(0.2)

    ticker_news_data_df_param.set_index(["Ticker", "Timestamp", "HeadlineID"], inplace = True)
    ticker_news_data_df_param.sort_index(inplace = True, level = 0)
    ticker_news_data_df_param.sort_index(inplace = True, level = 1, ascending = False, sort_remaining = False)
    return ticker_news_data_df_param


# ### **Main program used to collectively execute the whole script and retrieve the API key used for the inference package.**

# In[ ]:


def main():
    #startTime = datetime.now()
    
    with open("api_keys/huggingface.txt", "r") as f:
        huggingface_api_key = f.readline().strip()

    collection_start_date = date(2026, 4, 20) - relativedelta(years = 5)
    collection_end_date = date(2026, 4, 20)

    headline_data_df = get_data("tickers.txt", collection_start_date, collection_end_date, article_limit_per_day = 25)
    ticker_news_data_df_with_probabilities = add_probabilities(headline_data_df, huggingface_api_key)
    ticker_news_data_df_with_probabilities.to_parquet("5_year_ticker_headlines_with_finbert_rating.parquet")
        

    #print(datetime.now() - startTime)
    
main()

