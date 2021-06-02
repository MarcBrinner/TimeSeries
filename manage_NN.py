import numpy as np
import datasets
import load_data
import multiprocessing
import extract_data
import RBF_Approach
import random
from PIL import Image
from numba import njit
from tqdm import tqdm

train_years = range(2005, 2018)
test_years = range(2018, 2021)
momentum_stats_used = [((0, 0, 1), 0, "median"), ((0, 0, 2), 0, "median"), ((0, 0, 4), 1, "median"), ((0, 0, 7), 1, "median"),  ((0, 0, 10), 1, "median"),
                       ((0, 0, 15), 3, "median"), ((0, 1, 0), 3, "median"), ((0, 3, 0), 4, "median"), ((0, 7, 0), 5, "median"), ((1, 0, -7), 7, "median")]

@njit()
def calculate_earnings(number_of_stocks_to_select, earnings_daily, earnings_overall, number_of_days, current_stocks, number_of_symbols, stock_values):
    for i in range(number_of_days):
        highest_values = []
        for j in range(number_of_stocks_to_select):
            highest_value = -1
            highest_index = -1
            for k in range(number_of_symbols):
                if k not in highest_values and (highest_index == -1 or stock_values[i][k] > highest_value):
                    highest_value = stock_values[i][k]
                    highest_index = k
            highest_values.append(int(highest_index))
        min_val = stock_values[i][highest_values[-1]]
        max_val = stock_values[i][highest_values[0]] - min_val
        sum_values = 0
        for val in highest_values:
            stock_values[i][val] = (stock_values[i][val]-min_val+0.00000001)/(max_val+0.0000001) + 1
            sum_values += stock_values[i][val]
        current_stocks = np.zeros(number_of_symbols)
        for j in highest_values:
            current_stocks[j] = stock_values[i][j]/sum_values
        earnings_overall = earnings_overall * np.dot(current_stocks, earnings_daily[i])
    return earnings_overall


def evaluate_parameters(data, years, score_model, number_of_stocks_to_select=None):
    number_of_stocks_to_select = [2, 3, 4, 5, 6] if number_of_stocks_to_select is None else number_of_stocks_to_select
    for number in number_of_stocks_to_select:
        earnings_list = []
        for year in years:
            stock_values = np.squeeze(score_model.predict(np.asarray([data[year]["momentum"]]))[0], axis=-1)
            number_of_symbols = data[year]["number_of_symbols"]
            earnings_current_year = calculate_earnings(min(number, number_of_symbols), data[year]["earnings"], 1, data[year]["trading_days_current_year"],
                                                       np.zeros(number_of_symbols), number_of_symbols, stock_values)
            earnings_list.append(earnings_current_year)
        print(f"Evaluation for {number} stocks: {np.mean(earnings_list)}")

def compute_data_multiprocess(result_queue, data, symbols_and_changes_over_time, years):
    values = {}
    for year in years:
        new_data = extract_data.get_momentum_data_for_year(data, year, momentum_stats_used)
        values[year] = extract_data.extract_relevant_data_as_array(new_data, symbols_and_changes_over_time, year, values=["momentum", "earnings"])
    result_queue.put(values)

def prepare_data_for_NN(years):
    dataset = datasets.load_daily_data_indexed_by_years()
    symbols_and_changes_over_time = load_data.get_symbols_and_changes_over_time()
    number_of_processes = 8
    bar = tqdm(total=2*number_of_processes, position=0, leave=True, desc="Processing data")

    process_year_dict = {i: list() for i in range(number_of_processes)}
    i = 0
    while years:
        y = years.pop()
        process_year_dict[i].append(y)
        i = (i + 1) % number_of_processes
    result_queue = multiprocessing.Queue()
    processes = []
    for i in range(number_of_processes):
        new_dict = {symbol: dict() for symbol in dataset.keys()}
        years = process_year_dict[i]
        for year in years:
            for symbol in new_dict.keys():
                if year in dataset[symbol].keys() and year - 1 in dataset[symbol].keys():
                    if year not in new_dict[symbol].keys():
                        new_dict[symbol][year] = dataset[symbol][year]
                    if year - 1 not in new_dict[symbol].keys():
                        new_dict[symbol][year - 1] = dataset[symbol][year - 1]
        x = multiprocessing.Process(target=compute_data_multiprocess,
                                    args=(result_queue, new_dict, symbols_and_changes_over_time, years))
        processes.append(x)
        x.daemon = True
        x.start()
        bar.update(1)

    data = {}
    counter = 0
    while counter < number_of_processes:
        bar.update(1)
        new_data = result_queue.get(block=True)
        data = {**data, **new_data}
        counter += 1
    bar.close()
    return data

def create_images(score_model, year, data):
    score_model = NN_New.score_output_NN(score_model)
    scores = score_model.predict([np.asarray([data[year]["momentum"]]), np.asarray([data[year]["earnings"]])])[0]
    scores = scores - np.min(scores)
    scores_new = ((scores/np.max(scores))*255).astype(np.uint8)
    image = Image.fromarray(scores_new)
    image.show()
    image.save(f"images/{year}_all.jpg")

    shape = np.shape(scores)
    for i in range(shape[0]):
        sorted = np.sort(scores[i])
        third = sorted[-3]
        for j in range(shape[1]):
            if scores[i][j] < third:
                scores[i][j] = 0
            else:
                scores[i][j] = 255
    scores_new = ((scores / np.max(scores)) * 255).astype(np.uint8)
    image = Image.fromarray(scores_new)
    image.show()
    image.save(f"images/{year}_top_3.jpg")

if __name__ == '__main__':
    data = prepare_data_for_NN(list(range(2005, 2021)))
    cluster_centers = RBF_Approach.load_parameters(clusters=True, index=6)["clusters"]

    import NN_New
    score_model = NN_New.RBF_Model_2(cluster_centers, 10, 64)
    score_model.trainable = True
    complete_model = NN_New.train_over_whole_year_model(score_model)

#    print(complete_model.predict(list(map(lambda x: np.asarray([x]), data[2018]))))
    evaluate_parameters(data, range(2018, 2021), score_model)
    print(np.mean([complete_model.predict([np.asarray([data[year]["momentum"]]), np.asarray([data[year]["earnings"]])]) for year in range(2005, 2021)]))
    print(complete_model.predict([np.asarray([data[2010]["momentum"]]), np.asarray([data[2010]["earnings"]])]))
    for epoch in tqdm(range(1400)):
        years = list(range(2005, 2018))
        random.shuffle(years)
        for year in years:
            complete_model.fit([np.asarray([data[year]["momentum"]]), np.asarray([data[year]["earnings"]])], [np.asarray([3.0])], epochs=1, batch_size=1, verbose=0)
    print(np.mean([complete_model.predict([np.asarray([data[year]["momentum"]]), np.asarray([data[year]["earnings"]])]) for year in range(2005, 2021)]))
    print(complete_model.predict([np.asarray([data[2010]["momentum"]]), np.asarray([data[2010]["earnings"]])]))
    evaluate_parameters(data, range(2018, 2021), score_model)

    create_images(score_model, 2016, data)
    create_images(score_model, 2019, data)
    create_images(score_model, 2010, data)
