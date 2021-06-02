import datetime
from collections import Counter

def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

def round_to_two(number):
    return int(round(number*100))/100

def extract_symbols_end_of_year(symbols_over_time, year):
    if year < 2002:
        print("Too early!")
        return []
    if symbols_over_time[year-1][12][31]["has_data"]:
        symbols = symbols_over_time[year-1][12][31]["symbols"]
    else:
        prev_date = symbols_over_time[year-1][12][31]["prev"]
        symbols = symbols_over_time[year-1-prev_date[0]][12-prev_date[1]][31-prev_date[2]]["symbols"]
    return symbols

def get_most_similar_date(dataframe, date):
    year, month, day = date
    index = dataframe.index.get_loc(datetime.datetime(year, month, day), method='nearest')
    closest_date = dataframe.index[index]
    return index, closest_date