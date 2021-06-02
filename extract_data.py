import utils
import numpy as np
import load_data
from finance_operations import get_adjusted_values

def calculate_momentum_values(data, start_index, i, one_month_length, momentum_stats_used):
    stock_values = []
    current_price = data[start_index + i]["open"]
    for v in momentum_stats_used:
        index_shift = v[0]
        current_index = start_index + i - start_index * index_shift[0] - one_month_length * index_shift[1] - index_shift[2]
        range_val = v[1]
        stat = v[2]

        if range_val == 0:
            try:
                stock_values.append(data[current_index][stat])
            except Exception:
                if stat == "median":
                    stock_values.append((data[current_index]["high"] + data[current_index]["low"]) / 2)
                else:
                    raise Exception("Stat not known!")

        else:
            average_median = sum([data[current_index + i]["high"] + data[current_index + i]["low"]
                                  for i in range(-range_val, range_val + 1)]) / (2 * (2 * range_val + 1))
            stock_values.append(average_median)

    momentum_values = [(current_price / x - 1) for x in stock_values]
    return np.asarray(momentum_values)

def get_momentum_data_for_year(data, year, momentum_stats_used=None, normalize_length=False, norm_vals=None):
    momentum_dict = {s: dict() for s in data.keys()}
    trading_days_prev_year, trading_days_current_year = (
        utils.most_common([len(data[s][year - 1]) for s in data.keys() if year - 1 in data[s].keys()]),
        utils.most_common([len(data[s][year]) for s in data.keys() if year in data[s].keys()]))

    one_month_length = int(trading_days_prev_year / 12)
    for symbol in data.keys():
        if year - 1 not in data[symbol].keys() or year not in data[symbol].keys() or \
                len(data[symbol][year]) != trading_days_current_year or len(
            data[symbol][year - 1]) != trading_days_prev_year:
            continue
        new_data_dict = get_adjusted_values(data[symbol][year - 1] + data[symbol][year],
                                                               basepoint=trading_days_prev_year)
        momentum_list = []
        for i in range(trading_days_current_year):
            open_price = new_data_dict[trading_days_prev_year + i]["open"]
            if momentum_stats_used:
                momentum_values = calculate_momentum_values(new_data_dict, trading_days_prev_year, i,
                                                                    one_month_length, momentum_stats_used)
            else:
                momentum_values = []
            momentum_list.append({"date": new_data_dict[trading_days_prev_year + i]["date"],
                                  "earnings": new_data_dict[trading_days_prev_year + i+1]["open"]/open_price if i < trading_days_current_year-1
                                  else new_data_dict[trading_days_prev_year + i]["close"]/open_price,
                                  "momentum": momentum_values if not norm_vals else (np.asarray(momentum_values)-norm_vals[0])/norm_vals[1],
                                  "close": new_data_dict[trading_days_prev_year + i]["close"],
                                  "open": open_price})
        if normalize_length:
            for i in range(253-trading_days_current_year):
                new_val = momentum_list[-1].copy()
                new_val["earnings"] = 1
                momentum_list.append(new_val)

        momentum_dict[symbol][year] = momentum_list
    return momentum_dict

def extract_relevant_data_as_array(data, symbols_and_changes_over_time, year, values=None, only_s_and_p=True):
    if values is None:
        return
    symbols_and_changes_over_time = symbols_and_changes_over_time if symbols_and_changes_over_time is not None else load_data.get_symbols_and_changes_over_time()

    if only_s_and_p:
        possible_symbols = [s for s in utils.extract_symbols_end_of_year(symbols_and_changes_over_time, year)
                        if s in data.keys() and year in data[s].keys()]
    else:
        possible_symbols = [s for s in data.keys() if year in data[s].keys()]

    trading_days_current_year = utils.most_common([len(data[s][year]) for s in possible_symbols])

    symbols = [s for s in possible_symbols if len(data[s][year]) == trading_days_current_year]

    return_dict = {}
    for value in values:
        return_dict[value] = np.asarray([[data[symbol][year][i][value] for symbol in symbols] for i in range(trading_days_current_year)])

    return_dict["trading_days_current_year"] = trading_days_current_year
    return_dict["number_of_symbols"] = len(symbols)

    return return_dict