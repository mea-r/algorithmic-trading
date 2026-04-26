#!/usr/bin/env python
# coding: utf-8

# ### **Import libraries**: 
# - **Hugging Face** is a platform and library for utilizing machine learning models.
# - **InferenceClient** is a library available on Hugging Face and is primarily used to perform inference – the process of reaching conclusions given a set of inputs.
# - **Transformers** is a Python library created by Hugging Face that allows one to download, manipulate, and run thousands of pretrained, open-source AI models.
# - **Pipeline** is a package within Transformers that simplifies the process of working with advanced AI models.

# In[4]:


import requests
from huggingface_hub import InferenceClient
from transformers import pipeline
import pandas as pd
from datetime import datetime, timedelta, date
import fastparquet
import time
import sys
import traceback


# ### **Utilize FinBERT - a pre-trained NLP model specialized for financial sentiment analysis – adapted by ProsusAI to analyze the probabilities (positive/negative/neutral) associated with each headline.**

# In[5]:


def add_probabilities(ticker_news_data_df_param, api_key_param, all_metadata):
    pipe = pipeline("text-classification", model="ProsusAI/finbert")
    client = InferenceClient(provider="hf-inference", api_key=api_key_param)
    ticker_news_data_df_param[["pos", "neu", "neg"]] = 0.0, 0.0, 0.0
    ticker_news_data_df_param["Timestamp"] = pd.to_datetime(ticker_news_data_df_param["Timestamp"])
    rows_num, row_counter = len(ticker_news_data_df_param), 1

    for index, row in ticker_news_data_df_param.iterrows():
        headline = row["Headline"]
        grade = "ungraded"
        fail_count = 0

        while grade == "ungraded" or grade == "fail":
            try:
                result = client.text_classification(headline, model="ProsusAI/finbert")
                grade = "pass"
                
            except Exception as e:
                grade = "fail"
                fail_count += 1
                time_now = datetime.now()
                string = time_now.strftime("%d-%m-%Y") + " at " + time_now.strftime("%H:%M:%S")
                
                with open("phase2_error_log.txt", "a") as f:
                    f.write("\n \n")
                    f.write(f"Traceback information ({string}, fail count: {fail_count}): \n")
                    f.write(f"Ticker: {row["Ticker"]} \n")
                    f.write(f"Headline: {headline} \n")
                    f.write(f"Headline ID: {row["HeadlineID"]} \n")
                    f.write(f"Publication date: {row["Timestamp"]} \n")
                    traceback.print_exc(file=f)
                    

                match fail_count:
                    case 1:
                        time.sleep(30)
                    case 2:
                        time.sleep(60)
                    case 3:
                        time.sleep(180)
                    case 4:
                        time.sleep(300)
                    case 5:
                        time.sleep(600)
                    case _:
                        with open("phase2_error_log.txt", "a") as f:
                            f.write("Error persisted after 19 minutes. Script failed gracefully.")
                        sys.exit()

        for probability in result:
            row = row.copy()
            match probability["label"]:
                case "positive":
                    ticker_news_data_df_param.loc[index, "pos"] = probability.score
                case "neutral":
                    ticker_news_data_df_param.loc[index, "neu"] = probability.score
                case "negative":
                    ticker_news_data_df_param.loc[index, "neg"] = probability.score
        
        with open("status.txt", "w") as f:
            f.writelines(all_metadata)
            f.write(str(round((row_counter/rows_num)*100, 2)) + "% complete." + "\n")

        row_counter +=1
        time.sleep(0.2)

    ticker_news_data_df_param.set_index(["Ticker", "Timestamp", "HeadlineID"], inplace = True)
    ticker_news_data_df_param.sort_index(inplace = True, level = 0)
    ticker_news_data_df_param.sort_index(inplace = True, level = 1, ascending = False, sort_remaining = False)
    return ticker_news_data_df_param


# ### **Main program used to collectively execute the whole script and retrieve the API key used for the inference package.**

# In[6]:


def main():

    open("phase2_error_log.txt", "w").close()
    
    with open("status.txt", "r+") as f:
        content = f.read()
        time_now = datetime.now()
        start_string = "Phase 2 started on " + time_now.strftime("%d-%m-%Y") + " at " + time_now.strftime("%H:%M:%S") + "." + "\n"
        f.write(start_string)
        f.seek(0)
        all_metadata = f.readlines()

    with open("api_keys/huggingface.txt", "r") as f:
        huggingface_api_key = f.readline().strip()

    headline_data_df = pd.read_parquet("5_year_ticker_headlines.parquet")

    ticker_news_data_df_with_probabilities = add_probabilities(headline_data_df, huggingface_api_key, all_metadata)
    ticker_news_data_df_with_probabilities.to_parquet("5_year_ticker_headlines_with_finbert_rating.parquet")

    with open("status.txt", "w") as f:
        f.writelines(all_metadata)
        time_now = datetime.now()
        end_string = "Phase 2 completed on " + time_now.strftime("%d-%m-%Y") + " at " + time_now.strftime("%H:%M:%S") + "." + "\n"
        f.write(end_string) 
        
main()

