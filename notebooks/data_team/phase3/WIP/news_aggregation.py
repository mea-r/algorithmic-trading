#!/usr/bin/env python
# coding: utf-8

# ### **Import libraries**: 
# - **UUID** is a library that provides a way to generate universally unique identifiers. In this script, it is used to create identifiers for headlines.
# - **Dateutil** is an extension library of the Datetime module. In this script, is is used to iterate over a time inverval and allow the algebraic manipulation of dates in terms of years.

# In[9]:


import requests
import uuid
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from dateutil.rrule import rrule, DAILY
import random
import fastparquet
import time
import traceback


# ### **Create a data frame with ticker, headlineID, headline, article publication date, and article source columns.** ###

# In[10]:


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

# In[11]:


def get_data(txt_file_containing_tickers, collection_start_date, collection_end_date, start_string, article_limit_per_ticker_per_day):
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
            
            time_from, time_to = day, day + timedelta(days = 1)
            time_from, time_to = time_from.strftime("%Y%m%dT%H%M"), time_to.strftime("%Y%m%dT%H%M")
            
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&time_from={time_from}&time_to={time_to}&limit={limit}&apikey={alpha_vantage_api_key}"
            
            grade = "ungraded"
            fail_count = 0

            while grade == "ungraded" or grade == "fail":
                try:
                    response = requests.get(url)
                    data = response.json()
                    news = data["feed"]
                    grade = "pass"
                    
                except Exception as e:
                    grade = "fail"
                    fail_count += 1
                    time_now = datetime.now()
                    string = time_now.strftime("%d-%m-%Y") + " at " + time_now.strftime("%H:%M:%S")
                    
                    with open("phase1_error_log.txt", "a") as f:
                        f.write("\n \n")
                        f.write(f"Traceback information ({string}, fail count: {fail_count}): \n")
                        f.write(f"Ticker: {ticker} \n")
                        f.write(f"Date: {day} \n")
                        f.write(f"Data: {data} \n")
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
                            with open("phase1_error_log.txt", "a") as f:
                                f.write("\nError persisted after 19 minutes. Script failed gracefully.")
                            sys.exit

            
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
                indexes_to_remove = random.sample(range(0, len(headlines_no_dupe)), len(headlines_no_dupe) - article_limit_per_ticker_per_day)
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

            with open("status.txt", "w") as f:
                f.write(start_string)
                f.write(str(round((counter/total)*100, 2)) + "% complete." + "\n")
            
            counter +=1
            time.sleep(0.6)

    headline_data_packaged = package(headline_data)

    headline_data_packaged["Timestamp"] = pd.to_datetime(headline_data_packaged["Timestamp"])

    headline_data_packaged = headline_data_packaged.reset_index(drop=True)

    return headline_data_packaged


# ### **Main program used to collectively execute the whole script and retrieve the API key used for the inference package.**

# In[12]:


def main():

    open("phase1_error_log.txt", "w").close()

    with open("status.txt", "w") as f:
        time_now = datetime.now()
        start_string = "Phase 1 started on " + time_now.strftime("%d-%m-%Y") + " at " + time_now.strftime("%H:%M:%S") + "." + "\n"
        f.write(start_string) 

    collection_start_date = date(2026, 4, 23) - relativedelta(years = 5)
    collection_end_date = date(2026, 4, 23)

    headline_data_df = get_data("tickers.txt", collection_start_date, collection_end_date, start_string, article_limit_per_ticker_per_day = 25)

    headline_data_df.to_parquet("5_year_ticker_headlines.parquet")

    with open("status.txt", "w") as f:
        time_now = datetime.now()
        f.write(start_string)
        end_string = "Phase 1 completed on " + time_now.strftime("%d-%m-%Y") + " at " + time_now.strftime("%H:%M:%S") + "." + "\n"
        f.write(end_string)
 
main()

