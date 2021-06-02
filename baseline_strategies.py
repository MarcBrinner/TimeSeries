import datasets
import load_data
import finance_operations
import utils
import numpy as np

def equal_weight_strategy_test(test_year=2017):
    data = datasets.load_daily_data_indexed_by_years()
    symbols_and_changes_over_time = load_data.get_symbols_and_changes_over_time()
    if test_year < 2002:
        print("Starting year too early!")
        quit()

    symbols = [s for s in utils.extract_symbols_end_of_year(symbols_and_changes_over_time, test_year)
               if s in data.keys() and test_year in data[s].keys()]

    trading_days_current_year = utils.most_common([len(data[s][test_year]) for s in symbols])

    symbols = [s for s in symbols if len(data[s][test_year]) == trading_days_current_year]

    new_data_dict = {}
    for symbol in symbols:
        new_data_dict[symbol] = finance_operations.get_adjusted_values(data[symbol][test_year])

    initial_prizes = [new_data_dict[symbol][0]["open"] for symbol in symbols]
    end_prices = [new_data_dict[symbol][-1]["close"] for symbol in symbols]

    earnings = [x[1]/x[0] for x in zip(initial_prizes, end_prices)]
    earnings = np.mean(earnings) - 1

    print(f"The equal weight strategy yielded {earnings*100:1.2f}% from beginning of {test_year} to the end of {test_year}.")
    return earnings

def equal_weight_strategy_US_multiyear_test(test_year_start=2017, test_year_end=None):
    test_year_end = test_year_end if test_year_start is not None else test_year_start

    if test_year_start < 2002:
        print("Starting year too early!")
        quit()

    earnings = []
    for year in range(test_year_start, test_year_end+1):
        change = equal_weight_strategy_test(year)
        earnings.append(change)
    earnings = sum(earnings)/len(earnings)
    if test_year_start != test_year_end:
        print(f"The equal weight strategy yielded on average {earnings*100:1.2f}% per year from beginning of {test_year_start} to the end of {test_year_end}.")

def index_fund_strategy_test(fund="US", test_year_start=2017, test_year_end=None):
    if test_year_end is None:
        test_year_end = test_year_start
    data = datasets.load_daily_data_index_fund(fund)

    earnings = []
    for year in range(test_year_start, test_year_end+1):
        if year not in data.keys():
            print("No data available for some of the timeframe.")
            return
        start_price = data[year][0][0]
        end_price = data[year][-1][1]
        earnings.append(end_price/start_price - 1)
    earnings = np.mean(earnings)
    print(f"The index fund strategy yielded {earnings * 100:1.2f}% from beginning of {test_year_start} to the end of {test_year_end}.")

if __name__ == '__main__':
    index_fund_strategy_test("US", 2018, 2020)
    equal_weight_strategy_US_multiyear_test(2018, 2020)