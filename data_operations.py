import pandas as pd
import matplotlib.pyplot as plt

def convert_dict_to_dataframe(dict: dict):
    all_items = []
    for year in dict.keys():
        all_items = all_items + dict[year]
    dataframe = pd.DataFrame([[float(item["open"]), float(item["high"]), float(item["low"]), float(item["close"]), float(item["adj. close"]),
                               float(item["split"]), float(item["dividend"]), item["date"]] for item in all_items],
                                 columns=["open", "high", "low", "close", "adj. close", "split", "dividend", "date"])
    dataframe["date"] = pd.to_datetime(dataframe["date"])
    dataframe.set_index("date", inplace=True)
    return dataframe

def plot_dataframe(dataframe, value="close"):
    dataframe[value].plot()
    plt.show()