#!/usr/bin/env python
# coding: utf-8

# ### **Import libraries**: 

# In[1]:


import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import sys
import fastparquet
import warnings
from IPython.display import clear_output


# ### **Load Parquet file into a Pandas DataFrame object and calculate mean sentiment, sentiment standard deviation, number of articles, percentage of strongly negative articles, and sentiment momentum.**

# In[2]:


def load_and_calculate(parquet_file_name_param, all_metadata_param):
    df_ratings = pd.read_parquet(parquet_file_name_param)
    tickers = list(set(df_ratings.index.get_level_values(0)))
    tickers.sort()

    df_alt_features = pd.DataFrame()

    counter = 0
    total = len(df_ratings)

    for ticker in tickers:
        df_dates = pd.to_datetime(df_ratings.loc[ticker].index.get_level_values(0))
        df_dates = list(set([date.date().strftime("%Y-%m-%d") for date in df_dates]))
        df_dates.sort(reverse=True)
        for date in df_dates:
            df_temp = df_ratings.loc[ticker].xs(date, level = 0, drop_level = False)
            
            sent_diff = df_temp["pos"].sub(df_temp["neg"])
            sent_mean = sent_diff.values.mean()
            sent_std = sent_diff.std()
            news_count = len(sent_diff)

            if news_count == 1:
                sent_std = 0
            
            pct_strong_negative = round((len(df_temp[df_temp["neg"] > 0.7]) / len(df_temp))*100,2)
            try:
                sent_5_day_avg = np.mean([np.mean(df_ratings.loc[ticker].xs((pd.Timestamp(date) - timedelta(days = i)).strftime("%Y-%m-%d"), level = 0, drop_level = False)["pos"].sub(df_ratings.loc[ticker].xs((pd.Timestamp(date) - timedelta(days = i)).strftime("%Y-%m-%d"), level = 0, drop_level = False)["neg"])) for i in range(5)])
                sent_momentum = sent_mean - sent_5_day_avg
            except:
                sent_5_day_avg = np.nan
                sent_momentum = np.nan

            df_alt_features = pd.concat([df_alt_features, pd.DataFrame(data = {"Ticker": ticker, "Date": date, "sent_mean" : [sent_mean], "sent_std" : [sent_std], "news_count": [news_count], "pct_strong_negative": [pct_strong_negative], "sent_momentum": [sent_momentum]})])
            df_alt_features.dropna(inplace = True)

            with open("status.txt", "w") as f:
                f.writelines(all_metadata_param)
                f.write(str(round((counter/total)*100, 2)) + "% complete." + "\n")

            counter +=1
            
            
    df_alt_features = df_alt_features.reset_index(drop=True)
    df_alt_features["Date"] = pd.to_datetime(df_alt_features["Date"])
    df_alt_features.set_index(["Ticker", "Date"], inplace = True)
    df_alt_features.sort_index(inplace = True, level = 0, sort_remaining = False)

    return df_alt_features


# In[3]:


def main():

    with open("status.txt", "r+") as f:
        content = f.read()
        time_now = datetime.now()
        start_string = "Phase 3 started on " + time_now.strftime("%d-%m-%Y") + " at " + time_now.strftime("%H:%M:%S") + "." + "\n"
        f.write(start_string)
        f.seek(0)
        all_metadata = f.readlines()
    
    df_alt_features = load_and_calculate("5_year_ticker_headlines_with_finbert_rating.parquet", all_metadata)
    
    df_alt_features.to_parquet("5_year_ticker_headlines_with_finbert_rating_calculated.parquet")


    with open("status.txt", "w") as f:
        f.writelines(all_metadata)
        time_now = datetime.now()
        end_string = "Phase 3 completed on " + time_now.strftime("%d-%m-%Y") + " at " + time_now.strftime("%H:%M:%S") + "." + "\n"
        f.write(end_string) 

main()

