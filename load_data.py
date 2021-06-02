import pandas as pd
import requests
import json
import os
import time
import csv
import msgpack
import numpy as np
from collections import Counter
from tqdm import tqdm
from functools import lru_cache
from collections import defaultdict

key_file = open("API_Keys.txt", "r")
API_KEYS = json.loads(key_file.read())
API_KEY = API_KEYS[2]
key_file.close()

INVALID_CALL = "{\"Error Message\": \"Invalid API call. Please retry or visit the documentation (https://www.alphavantage.co/documentation/) for TIME_SERIES_DAILY_ADJUSTED.\"}"
TIME_LIMIT = "{\"Information\": \"Thank you for using Alpha Vantage! Our standard API call frequency is 5 calls per minute and 500 calls per day. Please visit https://www.alphavantage.co/premium/ if you would like to target a higher API call frequency.\"}"
TIME_LIMIT_2 = "{\"Note\": \"Thank you for using Alpha Vantage! Our standard API call frequency is 5 calls per minute and 500 calls per day. Please visit https://www.alphavantage.co/premium/ if you would like to target a higher API call frequency.\"}"
TIME_LIMIT_SENTENCE = "Thank you for using Alpha Vantage!"

def extract_all_historical_symbols_and_changes_from_file():
    rows = list(csv.reader(open(f'data/sp_500_stocks_history.csv')))[1:]
    d = defaultdict(lambda: {"has_prev": None, "has_next": None, "has_data": False, "symbols": None, "changes": None, "next": None, "prev": None})
    d[1]["has_prev"] = True
    print(d[1])
    quit()
    new_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"has_prev": None, "has_next": None, "has_data": False, "symbols": None, "changes": None, "next": None, "prev": None})))
    # new_dict = {}
    # for i in range(22):
    #     new_dict[2001+i] = {}
    #     for month in range(1, 13):
    #         new_dict[2001 + i][month] = {}
    #         for day in range(1, 32):
    #             new_dict[2001 + i][month][day] = {"has_prev": None, "has_next": None, "has_data": False, "symbols": None, "changes": None, "next": None, "prev": None}
    all_symbols = set()
    prev_symbols = []
    for row in rows:
        date = row[0]
        symbols = row[1].split(",")

        year = int(date[:4])
        month = int(date[5:7])
        day = int(date[8:10])

        if year < 2001:
            prev_symbols = symbols
            continue

        all_symbols.update(symbols)
        missing_symbols = [x for x in prev_symbols if x not in symbols]
        new_symbols = [x for x in symbols if x not in prev_symbols]
        changes_list = [(symbol, "del") for symbol in missing_symbols] + [(symbol, "add") for symbol in new_symbols]
        if len(changes_list) == 0:
            continue
        new_dict[year][month][day]["changes"] = changes_list
        new_dict[year][month][day]["has_data"] = True

        prev_symbols = symbols

        new_dict[year][month][day]["symbols"] = symbols

    day_counter = None
    month_counter = None
    year_counter = None
    for i in range(22):
        year = 2001+i
        if year_counter is not None:
            year_counter += 1
            month_counter -= 12
        for month in range(1, 13):
            if month_counter is not None:
                month_counter += 1
                day_counter -= 31
            for day in range(1, 32):
                if day_counter is not None:
                    day_counter += 1
                    new_dict[year][month][day]["prev"] = [year_counter, month_counter, day_counter]
                    new_dict[year][month][day]["has_prev"] = True
                else:
                    new_dict[year][month][day]["has_prev"] = False

                if new_dict[year][month][day]["has_data"]:
                    day_counter = 0
                    month_counter = 0
                    year_counter = 0

    day_counter = None
    month_counter = None
    year_counter = None
    for i in range(22):
        year = 2022 - i
        if year_counter is not None:
            year_counter += 1
            month_counter -= 12
        for j in range(0, 12):
            month = 12 - j
            if month_counter is not None:
                month_counter += 1
                day_counter -= 31
            for k in range(0, 31):
                day = 31 - k
                if day_counter is not None:
                    day_counter += 1
                    new_dict[year][month][day]["next"] = [year_counter, month_counter, day_counter]
                    new_dict[year][month][day]["has_next"] = True
                else:
                    new_dict[year][month][day]["has_next"] = False

                if new_dict[year][month][day]["has_data"]:
                    day_counter = 0
                    month_counter = 0
                    year_counter = 0

    all_symbols_list = open("data/all_S_and_P_symbols.txt", "w+")
    all_symbols_list.write(json.dumps(list(all_symbols)))
    all_symbols_list.close()

    new_json_file = open("data/S_and_P_constituents_and_changes_over_time.txt", "w+")
    new_json_file.write(json.dumps(new_dict))
    new_json_file.close()

@lru_cache(maxsize=1)
def get_symbols_and_changes_over_time():
    f = open("data/S_and_P_constituents_and_changes_over_time.txt", "r")
    symbols_and_changes_over_time = json.loads(f.read())
    f.close()
    new_dict = {}
    for year in symbols_and_changes_over_time.keys():
        new_dict[int(year)] = {}
        for month in symbols_and_changes_over_time[year].keys():
            new_dict[int(year)][int(month)] = {}
            for day in symbols_and_changes_over_time[year][month].keys():
                new_dict[int(year)][int(month)][int(day)] = symbols_and_changes_over_time[year][month][day]
    return new_dict

def get_daily_data(symbol):
    if os.path.isfile(f"data/TIME_SERIES_DAILY_ADJUSTED/{symbol}.txt"):
        file = open(f"data/TIME_SERIES_DAILY_ADJUSTED/{symbol}.txt")
        dict = json.loads(file.read())
        file.close()
        return dict
    elif os.path.isfile(f"data/TIME_SERIES_DAILY_ETFS/{symbol}.csv"):
        return list(csv.reader(open(f'data/TIME_SERIES_DAILY_ETFS/{symbol}.csv')))[1:]

def get_stock_symbols(symbols="US", load_not_available=False):
    if symbols == "US":
        stocks = json.loads(open("data/all_S_and_P_symbols.txt", "r").read())
        if not load_not_available:
            try:
                f = open("data/not_available_symbols.txt", "r+")
                not_available_symbols = json.loads(f.read())
                return [x for x in stocks if x not in not_available_symbols]
            except:
                pass
        return stocks
    if symbols == "US_current":
        symbols_over_time = get_symbols_and_changes_over_time()
        latest_year = max(list(symbols_over_time.keys()))
        latest_month = max(list(symbols_over_time[latest_year].keys()))
        latest_day = max(list(symbols_over_time[latest_year][latest_month].keys()))
        prev_values = symbols_over_time[latest_year][latest_month][latest_day]["prev"]
        return symbols_over_time[latest_year-prev_values[0]][latest_month-prev_values[1]][latest_day-prev_values[2]]["symbols"]



def load_not_available_symbols():
    f = open("data/not_available_symbols.txt", "r")
    try:
        not_available_symbols = json.loads(f.read())
    except:
        print("Could not load not available symbols.")
        not_available_symbols = []
    f.close()
    return not_available_symbols

def save_not_available_symbols(symbols):
    f = open("data/not_available_symbols.txt", "w+")
    f.write(json.dumps(symbols))
    f.close()

def download_dily_data(symbols="US"):
    method_string = "TIME_SERIES_DAILY_ADJUSTED"
    not_available_symbols = load_not_available_symbols()

    symbols = get_stock_symbols(symbols)
    for symbol in symbols:
        if symbol in not_available_symbols:
            continue
        if os.path.isfile(f"data/{method_string}/{symbol}.txt"):
            continue

        just_waited = False
        while True:
            data = requests.get(f"https://www.alphavantage.co/query?function={method_string}&symbol={symbol}&apikey={API_KEY}").json()
            string = json.dumps(data)
            if string == TIME_LIMIT or string == TIME_LIMIT_2:
                if just_waited:
                    print("Daily limit reached.")
                    quit()
                print("WAIT...")
                time.sleep(65)
                just_waited = True
            else:
                break

        if string == INVALID_CALL:
            print(f"Did not find {symbol}.")
            not_available_symbols.append(symbol)
            save_not_available_symbols(symbols)
            continue
        elif not string.startswith("{\"Meta Data\""):
            print(string)
            continue
        print(f"Downloaded {symbol}.")
        file = open(f"data/{method_string}/{symbol}.txt", "w+")
        file.write(string)
        file.close()
    print("Download done.")

def download_intraday_data():
    method_string = "TIME_SERIES_INTRADAY_EXTENDED"
    addition_string = "&interval=1min"

    not_available_symbols = load_not_available_symbols()

    symbols = get_stock_symbols(symbols="US_current")

    for symbol in symbols:
        for i in range(2):
            for j in range(12):
                if symbol in not_available_symbols:
                    continue
                if os.path.isfile(f"data/{method_string}/{symbol}_{i+1}_{j+1}.txt"):
                    continue

                just_waited = False
                while True:
                    string = requests.get(f"https://www.alphavantage.co/query?function={method_string}{addition_string}&symbol={symbol}&apikey={API_KEY}&slice=year{i+1}month{j+1}").text

                    if string.find(TIME_LIMIT_SENTENCE) >= 0:
                        if just_waited:
                            print("Daily limit reached.")
                            quit()
                        print("WAIT...")
                        time.sleep(65)
                        just_waited = True
                    else:
                        break

                if string == INVALID_CALL:
                    print(f"Did not find {symbol}.")
                    not_available_symbols.append(symbol)
                    save_not_available_symbols(not_available_symbols)
                    continue
                elif not string.startswith("time"):
                    print(string)
                    continue
                print(f"Downloaded {symbol}_{i+1}_{j+1}.")
                file = open(f"data/{method_string}/{symbol}_{i+1}_{j+1}.txt", "w+")
                cr = csv.reader(string.splitlines(), delimiter=',')
                my_list = list(cr)
                file.write(json.dumps(my_list))
                file.close()
    print("Download done.")

def process_intraday_files():
    files = os.listdir("data/TIME_SERIES_INTRADAY_EXTENDED")
    new_folder = "data/TIME_SERIES_INTRADAY"
    symbols = get_stock_symbols()
    all_days = {}
    for symbol in tqdm(symbols, desc="Processing intraday data", leave=True):
        if f"{symbol}_1_1.txt" in files:
            for i in range(2):
                for j in range(12):
                    with open(f"data/TIME_SERIES_INTRADAY_EXTENDED/{symbol}_{i + 1}_{j + 1}.txt", "r") as f:
                        current_list = json.loads(f.read())
                        day_data = {}
                        for row in current_list[1:]:
                            day = row[0][:10]
                            if day not in day_data:
                                day_data[day] = []
                            hour = int(row[0][11:13])
                            minute = int(row[0][14:16])
                            if hour > 15 or hour < 9 or (hour == 9 and minute < 31):
                                continue
                            day_data[day].append({"open": float(row[1]), "high": float(row[2]), "low": float(row[3]),
                                                  "close": float(row[4]), "time": row[0]})
                        for day in day_data.keys():
                            if day not in all_days:
                                all_days[day] = 1
                            else:
                                all_days[day] += 1
                            day_data[day].reverse()
                            with open(f"{new_folder}/{day}_{symbol}.msgpack", 'wb') as f:
                                f.write(msgpack.packb(day_data[day]))

def create_dataset_split_intraday():
    files = os.listdir("data/TIME_SERIES_INTRADAY")
    list = [file[:10] for file in files]
    counts = dict(Counter(list))
    all_days = sorted(counts.keys())
    feasible_days = [day for day in all_days if counts[day] > 300]
    number_of_splits = 4
    number_of_days_per_split = int(np.floor(len(feasible_days)/number_of_splits))
    splits = [i * number_of_days_per_split for i in range(number_of_splits)] + [len(feasible_days)]
    split_lists = [feasible_days[splits[i]:splits[i+1]] for i in range(number_of_splits)]
    with open("data/split_list.txt", "w+") as f:
        f.write(json.dumps(split_lists))

def load_splits_list():
    with open("data/split_list.txt", "r") as f:
        return json.loads(f.read())

if __name__ == '__main__':
    print(load_splits_list())
    quit()
    create_dataset_split_intraday()
