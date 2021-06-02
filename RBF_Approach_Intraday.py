import datasets
import load_data
import finance_operations
import numpy as np
import utils
import multiprocessing
import json
from numba import njit
from scipy import optimize
from sklearn.cluster import KMeans
from tqdm.auto import tqdm
from multiprocessing import Queue
from cmaes import CMA
from collections import Counter

parameter_index = 2
train_years = range(2005, 2018)
test_years = range(2018, 2021)
momentum_stats_used = [((0, 1), 0, "median"), ((0, 2), 0, "median"), ((0, 4), 1, "median"), ((0, 7), 1, "median"),  ((0, 10), 1, "median"),
                       ((0, 15), 2, "median"), ((0, 25), 2, "median"), ((0, 35), 2, "median"), ((0, 55), 4, "median"), ((1, 0), 0, "median"),
                       ((2, 0), 0, "median"), ((3, 0), 0, "median"), ((5, 0), 0, "median")]
cluster_numbers = 40
number_of_processes = 13

def fill_missing_values_with_linear_interpolation(data_intraday):
    for symbol in data_intraday.keys():
        if len(data_intraday[symbol]) > 389:
            print("What???")
        elif len(data_intraday[symbol]) < 360:
            continue
        else:
            missing_indices = []
            hour = 9
            minute = 30
            for i in range(389):
                minute = minute + 1 if minute < 59 else 0
                hour = hour if minute > 0 else hour + 1
                current_index = i - len(missing_indices)
                current_hour = int(data_intraday[symbol][current_index]["time"][11:13])
                current_minute = int(data_intraday[symbol][current_index]["time"][14:16])
                if current_hour == hour and current_minute == minute:
                    continue
                else:
                    missing_indices.append(i)
            if 0 in missing_indices:
                data_intraday[symbol].insert(0, data_intraday[symbol][0].copy())
                del missing_indices[0]
            for index in missing_indices:
                v_0 = data_intraday[symbol][index-1]
                v_1 = data_intraday[symbol][index]
                data_intraday[symbol].insert(index, {"open": (v_0["open"]+v_1["open"])/2, "close": (v_0["close"]+v_1["close"])/2,
                                                     "high": (v_0["high"]+v_1["high"])/2, "low": (v_0["low"]+v_1["low"])/2})

def adjust_daily_data(data, dates):
    new_data = {date: {} for date in dates}
    for date in dates:
        year = int(date[:4])
        for symbol in data.keys():
            if year not in data[symbol].keys(): continue
            index = ([i for i in range(len(data[symbol][year])) if data[symbol][year][i]["date"].startswith(date)]+[-1])[0]
            if index == -1: continue

            if diff := index - 5 < 0:
                if year-1 in data[symbol].keys():
                    datapoints = data[symbol][year-1][diff:] + datapoints[:index+1]
                else:
                    continue
            else:
                datapoints = data[symbol][year][index-5:index+1]

            new_datapoints = finance_operations.get_adjusted_values(datapoints, 5)
            new_data[date][symbol] = new_datapoints
    return new_data

def calculate_momentum_values_for_day(data_intraday, data_day):
    fill_missing_values_with_linear_interpolation(data_intraday)
    symbols = [s for s in data_intraday.keys() if len(data_intraday[s]) == 389 and s in data_day.keys()]
    data = {symbol: list() for symbol in symbols}

    for symbol in symbols:
        for t in range(60, 389):
            stock_values = []
            current_price = data_intraday[symbol][t]["open"]
            for v in momentum_stats_used:
                index_shift = v[0]
                range_val = v[1]
                stat = v[2]

                if index_shift[1] != 0:
                    current_index = t - index_shift[1]

                    if range_val == 0:
                        try:
                            stock_values.append(data_intraday[symbol][current_index][stat])
                        except Exception:
                            if stat == "median":
                                stock_values.append((data_intraday[symbol][current_index]["high"] + data_intraday[symbol][current_index]["low"]) / 2)
                            else:
                                raise Exception("Stat not known!")

                    else:
                        average_median = sum([data_intraday[symbol][current_index-i]["high"] + data_intraday[symbol][current_index+i]["low"]
                                              for i in range(-range_val, range_val + 1)]) / (2 * (2 * range_val + 1))
                        stock_values.append(average_median)
                else:
                    current_index = -index_shift[0]-1
                    if stat == "median":
                        average_median = sum([data_day[symbol][current_index - i]["high"] +
                                              data_day[symbol][current_index + i]["low"]
                                              for i in range(-range_val, range_val + 1)]) / (2 * (2 * range_val + 1))
                        stock_values.append(average_median)
            momentum_values = [(current_price / x - 1) for x in stock_values]
            data[symbol].append({"current price": current_price, "momentum": momentum_values})
    return data


def select_stocks(parameters, data, symbols, index, money):
    linear_classifier_coefficients = parameters[0][:cluster_numbers]
    sigma = parameters[0][cluster_numbers]
    number_of_stocks_to_select = min(len(symbols), int(round(parameters[1] * 510)))
    cluster_centers = parameters[2]

    stocks_with_values = []
    for symbol in symbols:
        current_price = data[symbol][index]["open"]
        statistics = data[symbol][index]["momentum"]

        value = np.dot(np.exp(-np.sum(np.square(cluster_centers - statistics), axis=-1)/sigma), linear_classifier_coefficients)
        stocks_with_values.append([symbol, np.exp(value), current_price])

    stocks_with_values.sort(key=lambda x: x[1])
    stocks_with_values = stocks_with_values[-number_of_stocks_to_select:]
    min_val = stocks_with_values[0][1]
    max_val = stocks_with_values[-1][1] - min_val
    stocks_with_values = [(s[0], (s[1]-min_val)/max_val+1, s[2]) for s in stocks_with_values]
    sum_exp_vals = sum([item[1] for item in stocks_with_values])
    current_stocks = [(item[0], (money/item[2])*(item[1]/sum_exp_vals)) for item in stocks_with_values[:number_of_stocks_to_select]]
    return current_stocks

def get_earnings_calculation_parameters(data, symbols_and_changes_over_time, year):
    symbols_and_changes_over_time = symbols_and_changes_over_time if symbols_and_changes_over_time is not None else load_data.get_symbols_and_changes_over_time()

    possible_symbols = [s for s in utils.extract_symbols_end_of_year(symbols_and_changes_over_time, year)
                        if s in data.keys() and year in data[s].keys()]

    trading_days_current_year = utils.most_common([len(data[s][year]) for s in possible_symbols])

    symbols = [s for s in possible_symbols if len(data[s][year]) == trading_days_current_year]

    new_data_dict = {}
    for symbol in symbols:
        new_data_dict[symbol] = data[symbol][year]

    new_momentum_values = np.asarray([[data[symbol][year][i]["momentum"] for symbol in symbols] for i in range(trading_days_current_year)])
    new_opening_values = np.asarray([[data[symbol][year][i]["open"] for symbol in symbols] for i in range(trading_days_current_year)])
    new_final_closing_values = np.asarray([data[symbol][year][trading_days_current_year-1]["close"] for symbol in symbols])

    return new_momentum_values, new_opening_values, new_final_closing_values, trading_days_current_year, len(symbols)

@njit()
def calculate_earnings(linear_classifier_coefficients, cluster_centers, sigma, number_of_stocks_to_select, money,momentum_values,
                       opening_prices, number_of_days, current_stocks, end_prices, number_of_symbols, stock_values, sorted_values):
    for i in range(number_of_days):
        if i != 0:
            money = np.dot(current_stocks, opening_prices[i])
        for j in range(number_of_symbols):
            stock_values[j] = np.exp(np.dot(np.exp(-np.sum(np.square(cluster_centers - momentum_values[i][j]), axis=-1) * sigma),linear_classifier_coefficients))
            sorted_values[j] = stock_values[j]
        highest_values = []
        for j in range(number_of_stocks_to_select):
            highest_value = -1
            highest_index = -1
            for k in range(number_of_symbols):
                if k not in highest_values and (highest_index == -1 or stock_values[k] > highest_value):
                    highest_value = stock_values[k]
                    highest_index = k
            highest_values.append(int(highest_index))
        min_val = stock_values[highest_values[-1]]
        max_val = stock_values[highest_values[0]] - min_val
        sum_exp_vals = 0
        for val in highest_values:
            stock_values[val] = (stock_values[val]-min_val)/max_val + 1
            sum_exp_vals += stock_values[val]
        current_stocks = np.zeros(number_of_symbols)
        for j in highest_values:
            current_stocks[j] = money/opening_prices[i][j] * stock_values[j]/sum_exp_vals
    money = np.dot(end_prices, current_stocks)
    return money

def multiprocess_wrapper(input_queue, result_queue, data, symbols_and_changes_over_time, years):
    input_values = {}
    for year in years:
        momentum_values = calculate_momentum_values_for_year(data, year)
        input_values[year] = get_earnings_calculation_parameters(momentum_values, symbols_and_changes_over_time, year)

    year_lists = {"train": [y for y in years if y in train_years], "test": [y for y in years if y in test_years]}
    result_queue.put(True)

    while True:
        try:
            input_data = input_queue.get(block=True, timeout=60)
        except:
            return

        parameters = input_data[0]
        cluster_centers = parameters[2]
        number_of_coefficients = np.shape(cluster_centers)[0]
        sigma = np.reciprocal(parameters[0][number_of_coefficients:])
        coefficients = parameters[0][:number_of_coefficients]

        for year in year_lists[input_data[1]]:
            earnings = calculate_earnings(coefficients, cluster_centers, sigma, min(input_values[year][4], parameters[1]),
                                          100_000, input_values[year][0], input_values[year][1], input_values[year][3],
                                          np.zeros(input_values[year][4]), input_values[year][2], input_values[year][4],
                                          np.zeros(input_values[year][4]), np.zeros(input_values[year][4])) / 100_000 - 1
            result_queue.put((year, earnings))

input_queue = Queue()
result_queue = Queue()
processes = list()
processes_started = False
test_year_data = {}

def start_processes():
    dataset = datasets.load_daily_data_indexed_by_years()
    symbols_and_changes_over_time = load_data.get_symbols_and_changes_over_time()

    test_years_list = list(test_years)
    train_years_list = list(train_years)
    process_year_dict = {i:list() for i in range(number_of_processes)}
    i = 0
    while test_years_list or train_years_list:
        if train_years_list:
            y = train_years_list.pop()
            process_year_dict[i].append(y)
        if test_years_list:
            y = test_years_list.pop()
            process_year_dict[i].append(y)
        i = (i+1)%number_of_processes

    for i in tqdm(range(number_of_processes), position=0, leave=True, desc="Starting worker processes"):
        new_dict = {symbol:dict() for symbol in dataset.keys()}
        years = process_year_dict[i]
        for year in years:
            for symbol in new_dict.keys():
                if year in dataset[symbol].keys() and year-1 in dataset[symbol].keys():
                    if year not in new_dict[symbol].keys():
                        new_dict[symbol][year] = dataset[symbol][year]
                    if year-1 not in new_dict[symbol].keys():
                        new_dict[symbol][year-1] = dataset[symbol][year-1]
        x = multiprocessing.Process(target=multiprocess_wrapper,
                                        args=(input_queue, result_queue, new_dict, symbols_and_changes_over_time, years))
        processes.append(x)
        x.daemon = True
        x.start()

    counter = 0
    while counter < number_of_processes:
        result_queue.get(block=True)
        counter += 1

    global processes_started
    processes_started = True

best_value = None
best_parameters = None
new_best = False

def evaluate_parameters_RBF_momentum(parameters, number_of_stocks, cluster_centers, type, print_values=False):
    parameters = [parameters, number_of_stocks, cluster_centers]
    if not processes_started:
        start_processes()
    for i in range(number_of_processes):
        input_queue.put((parameters, type))

    results = []
    while (type == "test" and len(results) < len(test_years)) or (type == "train" and len(results) < len(train_years)):
        results.append(result_queue.get(block=True, timeout=None))

    earnings = np.mean([x[1] for x in results])
    global best_value
    global best_parameters
    global new_best
    if type == "train" and (best_value is None or earnings > best_value):
        best_parameters = parameters[0]
        best_value = earnings
        new_best = True
        if print_values:
            print(f"The new best value is {best_value} with parameters {list(best_parameters)}")
    elif print_values:
        print(earnings)
    return -earnings

def optimize_RBF_momentum_parameters_CMAES(optimize_separate_sigma=False):
    cluster_centers = load_cluster_centers(0)
    num_coefficients = np.shape(cluster_centers)[0]
    num_sigmas = num_coefficients if optimize_separate_sigma else 1

    bounds = np.array([[-1, 1]]*num_coefficients + [[0.001, 10]]*num_sigmas)
    optimizer = CMA(mean=np.asarray([0]*num_coefficients + [0.5]*num_sigmas, dtype="float64"), sigma=0.2, population_size=140, bounds=bounds)
    #optimizer = CMA(mean=np.asarray(np.asarray(best_params), dtype="float64"), sigma=0.2, population_size=30, bounds=bounds)

    global best_parameters
    global best_value
    global new_best
    num_generations = 30
    best_eval = 0
    for generation in range(num_generations):
        solutions = []
        print(f"\nGeneration {generation + 1}/{num_generations}")

        for _ in tqdm(range(optimizer.population_size), desc="Function evaluations", leave=True, position=0):
            x = optimizer.ask()
            value = evaluate_parameters_RBF_momentum(x, 0.3, cluster_centers, "train")
            solutions.append((x, value))
        if new_best:
            best_eval = evaluate_parameters_RBF_momentum(best_parameters, 0.3, cluster_centers, "test")
        print(f"\nBest value: {best_value} with parameters: {list(best_parameters)}")
        print(f"Evaluation value: {best_eval}")
        optimizer.tell(solutions)

    print("\nFinding best selection hyperparameter...")
    best_earnings_selection_parameter = 0
    best_selection_parameter = None
    for value in np.linspace(0.01, 1, 50):
        earnings = -evaluate_parameters_RBF_momentum(best_parameters, value, cluster_centers, "train")
        earnings2 = -evaluate_parameters_RBF_momentum(best_parameters, value, cluster_centers, "test")
        print(f"{earnings} for {value} (train).")
        print(f"{earnings2} for {value} (test).")
        if best_selection_parameter is None or best_earnings_selection_parameter < earnings:
            best_earnings_selection_parameter = earnings
            best_selection_parameter = value

    print(f"\nThe best test value is {best_earnings_selection_parameter} for parameter {best_selection_parameter}.")

    evaluation_value = evaluate_parameters_RBF_momentum(best_parameters, best_selection_parameter, cluster_centers, "test")
    print(f"The strategy yields {evaluation_value} per year.")

    for process in processes:
        process.terminate()

    save_parameters(coefficients=best_parameters, selection_parameter=best_selection_parameter)

def find_k_means_clusters():
    dataset_generator = datasets.DailyDatasetManager(test_years=range(2018, 2022))
    dataset = dataset_generator.get_complete_data_batch(momentum_stats_used, ["momentum"])
    raw_values = [x["momentum"] for x in dataset]

    k_means = KMeans(n_clusters=20, verbose=1, n_init=20)
    k_means.fit(raw_values)
    cluster_centers = k_means.cluster_centers_

    random_points = np.random.choice(range(len(dataset)), 20, replace=False)
    all_cluster_centers = np.asarray(list(cluster_centers) + [raw_values[i] for i in random_points])

    indices = []
    for val in raw_values:
        index = np.argmin(np.sum(np.square(np.subtract(all_cluster_centers, np.asarray(val))), axis=-1))
        indices.append(index)
    print(Counter(indices))
    save_parameters(cluster_centers=cluster_centers)
    quit()

def optimize_RBF_momentum_parameters_BFGS(optimize_separate_sigma=True):
    parameters, cluster_centers, _ = load_parameters(2)
    num_coefficients = np.shape(cluster_centers)[0]
    num_sigmas = num_coefficients if optimize_separate_sigma else 1
    bounds = np.array([[-1, 1]]*num_coefficients + [[0.001, 10]]*num_sigmas)
    res = optimize.minimize(evaluate_parameters_RBF_momentum, np.asarray(list(np.random.randn(num_coefficients)*0.1) + [0.2]*num_sigmas) , args=(0.3, cluster_centers, "train", True), method="Powell", bounds=bounds)
    #res = optimize.minimize(evaluate_parameters_RBF_momentum, parameters , args=(0.3, cluster_centers, "train", True), method="Powell", options={'disp': True}, bounds=bounds)
    np.save(f"parameters/RBF_Evo/{parameter_index}_coefficients_BFGS.npy", res.x)

def load_parameters(index=parameter_index, BFGS=False):
    if BFGS:
        coefficients = np.load(f"parameters/RBF_Evo/{index}_coefficients_BFGS.npy")
    else:
        coefficients = np.load(f"parameters/RBF_Evo/{index}_coefficients.npy")
    selection_parameter = np.load(f"parameters/RBF_Evo/{index}_selection.npy")[0]
    cluster_centers = np.load(f"parameters/RBF_Evo/{index}_cluster_centers.npy")

    return coefficients, cluster_centers, selection_parameter

def load_cluster_centers(index=parameter_index):
    return np.load(f"parameters/RBF_Evo/{index}_cluster_centers.npy")

def save_parameters(coefficients=None, selection_parameter=None, cluster_centers=None, index=parameter_index):
    if coefficients is not None:
        np.save(f"parameters/RBF_Evo/{index}_coefficients.npy", coefficients)
    if selection_parameter:
        np.save(f"parameters/RBF_Evo/{index}_selection.npy", np.asarray([selection_parameter]))
    if cluster_centers is not None:
        np.save(f"parameters/RBF_Evo/{index}_cluster_centers.npy", cluster_centers)

def transfer_cluster_centers(origin, target):
    _, cluster_centers, _ = load_parameters(origin)
    save_parameters(cluster_centers=cluster_centers, index=target)

def evaluate_function(index=parameter_index, number_of_stocks_to_select=20, BFGS=False):
    parameters, cluster_centers, selection_parameter = load_parameters(index, BFGS)
    evaluate_parameters_RBF_momentum(parameters, number_of_stocks_to_select, cluster_centers, "test", True)

if __name__ == '__main__':
    daily_data = datasets.load_daily_data_indexed_by_years()
    dataset = datasets.load_intraday_data_dict(0)
    dates = list(dataset.keys())
    d = adjust_daily_data(daily_data, dates)
    date = dates[0]
    m = calculate_momentum_values(dataset[date], d[date])

    print()
    #evaluate_function(index=2, BFGS=True, number_of_stocks_to_select=3)
    quit()
    #parameters, cluster_centers, selection_parameter = load_parameters(1)
    #evaluate_parameters_RBF_momentum(parameters, 0.3, cluster_centers, "test", True)
    #quit()
    #quit()
    optimize_RBF_momentum_parameters_BFGS()
    #optimize_RBF_momentum_parameters_CMAES(True)
