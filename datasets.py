import load_data
import pandas as pd
import numpy as np
import random
import utils
import time
import msgpack
import os
from tqdm import tqdm
from functools import lru_cache
from multiprocessing import Pool, Process, Manager
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
pd.set_option("min_rows", 5411)
pd.set_option("max_rows", 5411)
pd.options.display.width = 0

index_funds = {"US": "SXR8.DE",
               "DE": "EXS1.DE",
               "WORLD": "EUNL.DE"}

class DailyDatasetManager():

    def __init__(self, test_years=None, all_years=None):
        self.test_years = test_years if test_years is not None else list(range(2018, 2022))
        self.all_years = all_years if all_years is not None else list(range(2001, 2022))
        self.data = load_daily_data_indexed_by_years()
        self.symbols = []
        self.probabilities_train = {}
        self.probabilities_test = {}
        self.symbol_years = {}
        self.year_lengths = {}

        self.create_symbol_information()
        self.create_year_length_statistics()

    def create_symbol_information_pd(self):
        self.symbols = list(self.data.keys())
        self.symbol_years = {}
        number_of_test_years = 0
        number_of_train_years = 0
        for symbol in tqdm(self.symbols, desc="Indexing data ", position=0, leave=True):
            years_train_current_symbol = []
            years_test_current_symbol = []
            for year in self.test_years:
                try:
                    year_data = self.data[str(year)]
                    years_test_current_symbol.append(year)
                    number_of_test_years += 1
                except Exception:
                    pass
            for year in [x for x in self.all_years if x not in self.test_years]:
                try:
                    year_data = self.data[str(year)]
                    years_train_current_symbol.append(year)
                    number_of_train_years += 1
                except Exception:
                    pass
            self.symbol_years[symbol] = {"test": years_test_current_symbol, "train": years_train_current_symbol}

        self.probabilities_train = {s:len(self.symbol_years[s]["train"])/number_of_train_years for s in self.symbols}
        self.probabilities_test = {s:len(self.symbol_years[s]["test"])/number_of_test_years for s in self.symbols}

    def create_symbol_information(self):
        self.symbols = list(self.data.keys())
        self.symbol_years = {}
        number_of_test_years = 0
        number_of_train_years = 0
        for symbol in self.symbols:
            self.symbol_years[symbol] = {}
            data_indexed_by_year = self.data[symbol]
            all_years_available = list(data_indexed_by_year.keys())
            years_train = {year:len(self.data[symbol][year]) for year in all_years_available if year not in self.test_years}
            years_test = {year:len(self.data[symbol][year]) for year in all_years_available if year in self.test_years}
            number_of_test_years += len(years_test)
            number_of_train_years += len(years_train)
            self.symbol_years[symbol]["train"] = years_train
            self.symbol_years[symbol]["test"] = years_test

        self.probabilities_train = {s: len(self.symbol_years[s]["train"]) / number_of_train_years for s in self.symbols}
        self.probabilities_test = {s: len(self.symbol_years[s]["test"]) / number_of_test_years for s in self.symbols}

    def create_year_length_statistics(self):
        lengths = {year:list() for year in self.all_years}
        for symbol in self.symbols:
            for year in self.all_years:
                try:
                    lengths[year].append(len(self.data[symbol][year]))
                except Exception:
                    pass
        self.year_lengths = {year:utils.most_common(lengths[year]) for year in self.all_years}

    def get_batch_symbols_and_indices(self, batch_size, furthest_back):
        batch = []
        while len(batch) < batch_size:
            symbol = np.random.choice(self.symbols, 1, p=self.probabilities_train)
            year = int(random.choice(self.symbol_years[symbol][0]))

            if year-1 in self.symbol_years[symbol][0] + self.symbol_years[symbol][1]:
                all_datapoints = list(self.data[symbol][year-1].values()) + list(self.data[symbol][year].values())
                index = random.randint(max(furthest_back, len(self.data[symbol][year-1])), len(all_datapoints)-1)
            else:
                if len(self.data[symbol][year]) < furthest_back:
                    continue
                all_datapoints = self.data[symbol][year]
                index = random.randint(furthest_back, len(self.data[symbol][year]))

    def get_random_data_batch(self, batch_size, values, ret, only_train=True ):
        batch = []
        symbols = list(self.probabilities_train.keys())
        probabilities = list(self.probabilities_train.values())
        random_symbols = list(np.random.choice(symbols, batch_size, p=probabilities))
        not_working_symbols = []
        while len(batch) < batch_size:
            try:
                symbol = random_symbols.pop()
            except Exception:
                probabilities = [p/sum(probabilities) for p in probabilities]
                random_symbols = list(np.random.choice(symbols, int(max(batch_size/5, 100)), p=probabilities))
                symbol = random_symbols.pop()

            if len(self.symbol_years[symbol]["train"]) <= 1 or symbol in not_working_symbols:
                continue

            all_year_lengths = {**self.symbol_years[symbol]["train"], **self.symbol_years[symbol]["test"]}
            possible_years = [year for year in self.symbol_years[symbol]["train"].keys()
                              if year in all_year_lengths.keys() and year in self.year_lengths.keys() and
                              (year - 1 in all_year_lengths.keys()) and (year - 1 in self.year_lengths.keys()) and
                              abs((all_year_lengths[year]/self.year_lengths[year])-1) < 0.02 and
                              abs((all_year_lengths[year-1]/self.year_lengths[year-1])-1) < 0.02 and
                              (not only_train or year not in self.test_years)]
            if len(possible_years) == 0:
                index = symbols.index(symbol)
                del symbols[index]
                del probabilities[index]
                not_working_symbols.append(symbol)
                continue

            year = int(random.choice(possible_years))
            year_length = len(self.data[symbol][year - 1])
            month_length = int(year_length / 12)
            index = year_length + random.randint(0, len(self.data[symbol][year])-1)

            current_data = self.data[symbol][year-1] + self.data[symbol][year]
            current_price =current_data[index]["open"]

            data_values = []
            for v in values:
                index_shift = v[0]
                current_index = index - year_length*index_shift[0] - month_length*index_shift[1] - index_shift[2]
                range_val = v[1]
                stat = v[2]

                if range_val == 0:
                    try:
                        data_values.append(current_data[current_index][stat])
                    except Exception:
                        if stat == "median":
                            data_values.append((current_data[current_index]["high"] + current_data[current_index]["low"]) /2)
                        else:
                            raise Exception("Stat not known!")

                else:
                    average_median = sum([current_data[current_index+i]["high"] + current_data[current_index+i]["low"]
                                          for i in range(-range_val, range_val+1)]) / (2*(2*range_val+1))
                    data_values.append(average_median)


            return_dict = dict()
            return_dict["current_price"] = current_price
            if "values" in ret:
                return_dict["values"] = data_values
            if "momentum" in ret:
                return_dict["momentum"] = [(current_price/x - 1) for x in data_values]
            return_dict["symbol"] = symbol
            return_dict["date"] = current_data[index]["date"]
            batch.append(return_dict)
        return batch

    def get_complete_data_batch(self, values, ret, only_train=True):
        batch = []
        symbols = list(self.probabilities_train.keys())
        for symbol in tqdm(symbols, desc="Creating dataset", leave=True, position=0):
            if len(self.symbol_years[symbol]["train"]) <= 1:
                continue

            all_year_lengths = {**self.symbol_years[symbol]["train"], **self.symbol_years[symbol]["test"]}
            possible_years = [year for year in all_year_lengths.keys()
                              if year in all_year_lengths.keys() and year in self.year_lengths.keys() and
                              (year - 1 in all_year_lengths.keys()) and (year - 1 in self.year_lengths.keys()) and
                              abs((all_year_lengths[year]/self.year_lengths[year])-1) < 0.02 and
                              abs((all_year_lengths[year-1]/self.year_lengths[year-1])-1) < 0.02 and
                              (not only_train or year not in self.test_years)]

            if len(possible_years) == 0:
                continue

            for year in possible_years:
                year_length = len(self.data[symbol][year - 1])
                month_length = int(year_length / 12)
                for base_index in range(0, len(self.data[symbol][year])):
                    index = year_length + base_index

                    current_data = self.data[symbol][year-1] + self.data[symbol][year]
                    current_price =current_data[index]["open"]

                    data_values = []
                    for v in values:
                        index_shift = v[0]
                        current_index = index - year_length*index_shift[0] - month_length*index_shift[1] - index_shift[2]
                        range_val = v[1]
                        stat = v[2]

                        if range_val == 0:
                            try:
                                data_values.append(current_data[current_index][stat])
                            except Exception:
                                if stat == "median":
                                    data_values.append((current_data[current_index]["high"] + current_data[current_index]["low"]) /2)
                                else:
                                    raise Exception("Stat not known!")

                        else:
                            average_median = sum([current_data[current_index+i]["high"] + current_data[current_index+i]["low"]
                                                  for i in range(-range_val, range_val+1)]) / (2*(2*range_val+1))
                            data_values.append(average_median)


                    return_dict = dict()
                    return_dict["current_price"] = current_price
                    if "values" in ret:
                        return_dict["values"] = data_values
                    if "momentum" in ret:
                        return_dict["momentum"] = [(current_price/x - 1) for x in data_values]
                    return_dict["symbol"] = symbol
                    return_dict["date"] = current_data[index]["date"]
                    batch.append(return_dict)
        return batch


@lru_cache(maxsize=1)
def load_daily_data_pd():
    symbols = load_data.get_stock_symbols()
    data = {}
    for symbol in tqdm(symbols, desc="Loading US S&P 500 daily data for 20 years ", position=0, leave=True):
        #symbol = "REGN"
        stock_data = load_data.get_daily_data(symbol)["Time Series (Daily)"]
        dataframe = pd.DataFrame([[key] + [float(x) for x in value.values()] for key, value in stock_data.items()],
                                 columns=["Date", "open", "high", "low", "close", "adj. close", "volume", "dividend", "split"])
        dataframe["Date"] = pd.to_datetime(dataframe["Date"])
        dataframe.set_index("Date", inplace=True)
        #print(dataframe)
        #data_operations.plot_dataframe(dataframe)

    return data

def load_daily_data_for_specific_symbol(symbol):
    stock_data = load_data.get_daily_data(symbol)["Time Series (Daily)"]
    stock_dict = {"symbol": symbol, "data": {}}
    for key, value in stock_data.items():
        year = int(key[:4])
        month = key[5:7]
        day = key[8:10]
        if year not in stock_dict["data"]:
            stock_dict["data"][year] = []
        stock_dict["data"][year].append(
            {"open": float(value["1. open"]), "close": float(value["4. close"]), "low": float(value["3. low"]),
             "high": float(value["2. high"]), "date": key, "split": float(value["8. split coefficient"]),
             "dividend": float(value["7. dividend amount"]), "adj. close": float(value["5. adjusted close"])})
    for year in stock_dict["data"].keys():
        stock_dict["data"][year].reverse()
    return stock_dict

@lru_cache(maxsize=1)
def load_daily_data_indexed_by_years():
    symbols = load_data.get_stock_symbols()
    pool = Pool(6)
    bar = tqdm(total=len(symbols), position=0, leave=True, desc="Loading daily data")
    files = pool.imap(load_daily_data_for_specific_symbol, symbols)
    data = {}
    for file in files:
        bar.update(1)
        data[file["symbol"]] = file["data"]
    pool.close()
    pool.join()
    bar.close()
    return data

def load_daily_data_index_fund(fund="US"):
    print(f"Loading {fund} index fund daily data for 20 years...", end="")
    symbol = index_funds[fund]
    stock_data = load_data.get_daily_data(symbol)

    stock_dict = {}
    for value in stock_data:
        year = int(value[0][:4])
        month = value[0][5:7]
        day = value[0][8:10]
        if year not in stock_dict:
            stock_dict[year] = []
        stock_dict[year].append((float(value[1]), float(value[4]), float(value[3]), float(value[2])))
    print("  Done.")
    return stock_dict

def load_intraday_data_for_day(day):
    files = os.listdir("data/TIME_SERIES_INTRADAY")
    files_to_load = [file for file in files if file.startswith(day)]
    data = {"date": day, "data": {}}
    for file in files_to_load:
        symbol = file[11:-8]
        f = open(f"data/TIME_SERIES_INTRADAY/{file}", "rb")
        data["data"][symbol] = msgpack.unpackb(f.read(), strict_map_key=False)
    return data


def load_intraday_data_dict(index):
    split = load_data.load_splits_list()[index]
    pool = Pool(8)
    bar = tqdm(total=len(split), leave=True, desc="Loading intraday data")
    files = pool.imap(load_intraday_data_for_day, split)
    data = {}
    for file in files:
        bar.update(1)
        data[file["date"]] = file["data"]
    pool.close()
    pool.join()
    bar.close()
    return data

if __name__ == '__main__':
    t = time.time()
    data = load_intraday_data_dict(0)
    print(time.time()-t)

    quit()
    load_daily_data_pd()
    quit()
    load_daily_data_indexed_by_years()