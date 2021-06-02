import datasets
import load_data
import numpy as np
import multiprocessing
import extract_data
from numba import njit
from scipy import optimize
from sklearn.cluster import KMeans
from tqdm.auto import tqdm
from multiprocessing import Queue
from cmaes import CMA
from collections import Counter

parameter_index = 3
train_years = range(2005, 2018)
test_years = range(2018, 2021)
momentum_stats_used = [((0, 0, 1), 0, "median"), ((0, 0, 2), 0, "median"), ((0, 0, 4), 1, "median"), ((0, 0, 7), 1, "median"),  ((0, 0, 10), 1, "median"),
                       ((0, 0, 15), 3, "median"), ((0, 1, 0), 3, "median"), ((0, 3, 0), 4, "median"), ((0, 7, 0), 5, "median"), ((1, 0, -7), 7, "median")]
cluster_numbers = 20
number_of_processes = 13
normalization_active = False

@njit()
def calculate_earnings(linear_classifier_coefficients, cluster_centers, sigma, number_of_stocks_to_select, momentum_values,
                       earnings_daily, number_of_days, current_stocks, number_of_symbols, stock_values):
    earnings_overall = 1
    for i in range(number_of_days):
        for j in range(number_of_symbols):
            stock_values[j] = np.exp(np.dot(np.exp(-np.sum(np.square(cluster_centers - momentum_values[i][j]), axis=-1) * sigma),linear_classifier_coefficients))
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
            current_stocks[j] = stock_values[j]/sum_exp_vals
        earnings_overall = earnings_overall * np.dot(current_stocks, earnings_daily[i])
    return earnings_overall

def multiprocess_wrapper(input_queue, result_queue, data, symbols_and_changes_over_time, years, norm):
    input_values = {}
    for year in years:
        new_data = extract_data.get_momentum_data_for_year(data, year, momentum_stats_used, norm_vals=norm)
        input_values[year] = extract_data.extract_relevant_data_as_array(new_data, symbols_and_changes_over_time, year, values=["momentum", "earnings"])

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
            earnings = calculate_earnings(coefficients, cluster_centers, sigma, int(min(input_values[year]["number_of_symbols"], parameters[1])),
                                          input_values[year]["momentum"], input_values[year]["earnings"],
                                          input_values[year]["trading_days_current_year"], np.zeros(input_values[year]["number_of_symbols"]),
                                          input_values[year]["number_of_symbols"], np.zeros(input_values[year]["number_of_symbols"]))
            result_queue.put((year, earnings))

input_queue = Queue()
result_queue = Queue()
processes = list()
processes_started = False

def start_processes(norm):
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
                                        args=(input_queue, result_queue, new_dict, symbols_and_changes_over_time, years, norm))
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

def evaluate_parameters_RBF_momentum(parameters, number_of_stocks, cluster_centers, type, norm=None, print_values=False):
    parameters = [parameters, number_of_stocks, cluster_centers]
    if not processes_started:
        start_processes(norm)
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
    parameters = load_parameters(clusters=True, normalization=normalization_active)
    if not normalization_active: parameters["norm"] = None
    cluster_centers = parameters["clusters"]
    print(cluster_centers)
    num_coefficients = np.shape(cluster_centers)[0]

    num_sigmas = num_coefficients if optimize_separate_sigma else 1

    bounds = np.array([[-1, 1]]*num_coefficients + [[0.001, 10]]*num_sigmas)
    optimizer = CMA(mean=np.asarray([0]*num_coefficients + [0.5]*num_sigmas, dtype="float64"), sigma=0.2, population_size=50, bounds=bounds)
    #optimizer = CMA(mean=np.asarray(np.asarray(best_params), dtype="float64"), sigma=0.2, population_size=30, bounds=bounds)

    global best_parameters
    global best_value
    global new_best
    num_generations = 4
    best_eval = 0
    for generation in range(num_generations):
        solutions = []
        print(f"\nGeneration {generation + 1}/{num_generations}")

        for _ in tqdm(range(optimizer.population_size), desc="Function evaluations", leave=True, position=0):
            x = optimizer.ask()
            value = evaluate_parameters_RBF_momentum(x, 150, cluster_centers, "train", parameters["norm"])
            solutions.append((x, value))
        if new_best:
            best_eval = -evaluate_parameters_RBF_momentum(best_parameters, 150, cluster_centers, "test", parameters["norm"])
            new_best = False
        print(f"\nBest value: {best_value} with parameters: {list(best_parameters)}")
        print(f"Evaluation value: {best_eval}")
        optimizer.tell(solutions)

    print("\nFinding best selection hyperparameter...")
    best_earnings_selection_parameter = 0
    best_selection_parameter = None
    for value in np.linspace(2, 150, 149):
        earnings = -evaluate_parameters_RBF_momentum(best_parameters, value, cluster_centers, "train", parameters["norm"])
        earnings2 = -evaluate_parameters_RBF_momentum(best_parameters, value, cluster_centers, "test", parameters["norm"])
        print(f"{earnings} for {value} (train).")
        print(f"{earnings2} for {value} (test).")
        if best_selection_parameter is None or best_earnings_selection_parameter < earnings:
            best_earnings_selection_parameter = earnings
            best_selection_parameter = value

    print(f"\nThe best test value is {best_earnings_selection_parameter} for parameter {best_selection_parameter}.")

    evaluation_value = -evaluate_parameters_RBF_momentum(best_parameters, best_selection_parameter, cluster_centers, "test", parameters["norm"])
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

def find_k_means_clusters_normalized():
    dataset_generator = datasets.DailyDatasetManager(test_years=range(2018, 2022))
    dataset = dataset_generator.get_complete_data_batch(momentum_stats_used, ["momentum"])
    raw_values = np.asarray([x["momentum"] for x in dataset])

    mean = np.sum(raw_values, axis=0)/len(raw_values)
    raw_values = raw_values - mean
    std = np.std(raw_values, axis=0)
    raw_values = raw_values/std

    normalization_data = np.asarray([mean, std])
    save_parameters(normalization=normalization_data)

    raw_values = np.asarray([x for x in raw_values if max(x) < 15])
    k_means = KMeans(n_clusters=30, verbose=1, n_init=20)
    pred = k_means.fit_predict(raw_values)
    cluster_centers = k_means.cluster_centers_

    print(Counter(pred))

    save_parameters(cluster_centers=cluster_centers)
    quit()

def train_SOM():
    dataset_generator = datasets.DailyDatasetManager(test_years=range(2018, 2022))
    dataset = dataset_generator.get_complete_data_batch(momentum_stats_used, ["momentum"])
    raw_values = np.asarray([x["momentum"] for x in dataset])

    som = SOM(m=8, n=8, dim=10)
    indices = som.fit_predict(raw_values)

    print(Counter(indices))
    cluster_centers = som.cluster_centers_
    print(cluster_centers)
    cluster_centers = np.reshape(cluster_centers, (np.shape(cluster_centers)[0]**2, np.shape(cluster_centers)[2]))
    save_parameters(cluster_centers=cluster_centers)
    quit()

def optimize_RBF_momentum_parameters_Powell(optimize_separate_sigma=True):
    parameters = load_parameters(index=parameter_index, coefficients=True, clusters=True, normalization=normalization_active)
    cluster_centers = parameters["clusters"]
    num_coefficients = np.shape(cluster_centers)[0]
    num_sigmas = num_coefficients if optimize_separate_sigma else 1
    bounds = np.array([[-1, 1]]*num_coefficients + [[0.001, 10]]*num_sigmas)
    res = optimize.minimize(evaluate_parameters_RBF_momentum, np.asarray(list(np.random.randn(num_coefficients)*0.1) + [0.2]*num_sigmas),
                            args=(150, cluster_centers, "train", parameters["norm"] if normalization_active else None, True), method="Powell", bounds=bounds)
    #res = optimize.minimize(evaluate_parameters_RBF_momentum, parameters, args=(150, cluster_centers, "train", parameters["norm"] if normalization_active else None, True),
    #                        method="Powell", options={'disp': True}, bounds=bounds)
    np.save(f"parameters/RBF_Evo/{parameter_index}_coefficients_BFGS.npy", res.x)

def load_parameters(index=parameter_index, BFGS=False, clusters=False, coefficients=False, selection_parameter=False, normalization=False):
    return_dict = {}
    if BFGS:
        params = np.load(f"parameters/RBF_Evo/{index}_coefficients_BFGS.npy")
        return_dict["BFGS"] = params
    if coefficients:
        params = np.load(f"parameters/RBF_Evo/{index}_coefficients.npy")
        return_dict["coef"] = params
    if selection_parameter:
        selection_parameter = np.load(f"parameters/RBF_Evo/{index}_selection.npy")[0]
        return_dict["selection"] = selection_parameter
    if clusters:
        centers = np.load(f"parameters/RBF_Evo/{index}_cluster_centers.npy")
        return_dict["clusters"] = centers
    if normalization:
        norm = np.load(f"parameters/RBF_Evo/{index}_normalization.npy")
        return_dict["norm"] = norm

    return return_dict

def save_parameters(coefficients=None, selection_parameter=None, cluster_centers=None, normalization=None, index=parameter_index):
    if coefficients is not None:
        np.save(f"parameters/RBF_Evo/{index}_coefficients.npy", coefficients)
    if selection_parameter:
        np.save(f"parameters/RBF_Evo/{index}_selection.npy", np.asarray([selection_parameter]))
    if cluster_centers is not None:
        np.save(f"parameters/RBF_Evo/{index}_cluster_centers.npy", cluster_centers)
    if normalization is not None:
        np.save(f"parameters/RBF_Evo/{index}_normalization.npy", normalization)

def evaluate_function(index=parameter_index, number_of_stocks_to_select=None, BFGS=False, scope="test"):
    number_of_stocks_to_select = [2, 3, 4, 5, 6] if number_of_stocks_to_select is None else number_of_stocks_to_select
    parameters = load_parameters(index=index, BFGS=BFGS, coefficients=not BFGS, clusters=True, normalization=normalization_active)
    for number in number_of_stocks_to_select:
        evaluate_parameters_RBF_momentum(parameters["BFGS"] if BFGS else parameters["coef"], number,
                                         parameters["clusters"], scope, norm=parameters["norm"] if normalization_active else None, print_values=True)

if __name__ == '__main__':
    evaluate_function(index=5, BFGS=False)
    quit()
    #optimize_RBF_momentum_parameters_Powell()
    optimize_RBF_momentum_parameters_CMAES()
