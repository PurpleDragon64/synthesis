# Scritp to analyze logs from value iteration

import pandas as pd

def global_stats(data):
    diff_ratio = data['diff count'] / data['value vector size']
    print('Pro každou iteraci poměr mezi počtem stavů u kterých se změnila hodnota oproti minulé iteraci ku všem stavům')
    print(diff_ratio.describe())

    diff_size = data['total diff'] / data['diff count']
    print('Pro každou iteraci průměrná velikost změny hodnoty stavu oproti minulé iteraci')
    print(diff_size.describe())

    diff_size_with_zeros = data['total diff'] / data['value vector size']
    print('Pro každou iteraci průměrná velikost změny hodnoty stavu oproti minulé iteraci se zahrnutím nulové změny')
    print(diff_size_with_zeros.describe())

def grouped_stats(data : pd.DataFrame):
    # should compensate different model sizes
    diff_ratios = []
    diff_sizes = []
    diff_sizes_with_zeros = []

    grouped = data.groupby('model')
    for model in grouped:
        diff_ratio = model[1]['diff count'] / model[1]['value vector size']
        diff_ratios.append(diff_ratio.mean())
        diff_size = model[1]['total diff'] / model[1]['diff count']
        diff_sizes.append(diff_size.mean())
        diff_size_with_zeros = model[1]['total diff'] / model[1]['value vector size']
        diff_sizes_with_zeros.append(diff_size_with_zeros.mean())

    new_data = pd.DataFrame({
        'diff_ratio' : diff_ratios,
        'diff_size' : diff_sizes,
        'diff_size_with_zeros' : diff_sizes_with_zeros
    })

    print(new_data.describe())




with open('log.csv') as f:
    data = pd.read_csv(f)

    print('------ global stats ---------')
    global_stats(data)
    print('------ grouped stats --------')
    grouped_stats(data)
