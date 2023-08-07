import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def func(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%\n({:d})".format(pct, absolute)
def create_graph(df, not_relevant_columns: list, save_dir: str):
    for column in df.columns:
        if column not in not_relevant_columns :
            value_counts = df[column].value_counts()
            y = np.array(value_counts.values)
            mylabels = [idx[::-1] for idx in value_counts.index]

            plt.pie(y, labels=mylabels, autopct = lambda pct: func(pct, y))
            plt.savefig(os.path.join(save_dir, column + '.png'))
            plt.show()
            plt.close()




def remove_incoret_pin(df, point_filed, good_val):
    return df[df[point_filed].str.contains(good_val) == True]


def get_duplicate_pin(df, pin_filed):
    ret_list = []
    pin_series = df[pin_filed].value_counts()
    for pin in pin_series.index:
        if pin_series[pin] > 1:
            ret_list.append(pin)
    return ret_list

def get_incorrect_pins(df, pin_filed, point_filed, good_val):
    correct = df[df[point_filed].str.contains(good_val) == True]
    return list(set(df[pin_filed]) - set(correct[pin_filed]))


def fusion_same_pin(df, pin_filed):
    aggregation_functions = {}
    for column in df.columns:
        if column == pin_filed:
            continue
        aggregation_functions[column] = 'last'

    f_new = df.groupby(df[pin_filed]).aggregate(aggregation_functions)
    return f_new

def analysis(csv_input: str, res_dir: str):
    pin_filed = 'סיסמה'
    point_filed = 'Score'#'ניקוד'
    good_val = '100 / 100'
    not_relevant_column = [pin_filed, point_filed, 'Timestamp']#'חותמת זמן']

    df = pd.read_csv(csv_input, sep=",", encoding='utf-8')
    print('incurrect pins:', get_incorrect_pins(df, pin_filed, point_filed, good_val))
    print('duplicate pin:', get_duplicate_pin(df, pin_filed))

    df = remove_incoret_pin(df, point_filed, good_val)
    df = fusion_same_pin(df, pin_filed)
    create_graph(df, not_relevant_column, res_dir)
    df.to_csv(os.path.join(res_dir, 'csv_output.csv'))

# def analysis_from_google_sheet(url, res_dir):
#     df = pd.read_csv(url, on_bad_lines='skip')
#     print('incurrect pins:', get_incorrect_pins(df))
#     print('duplicate pin:', get_duplicate_pin(df))
#     df = remove_incoret_pin(df)
#     df = fusion_same_pin(df)
#     df.to_csv(os.path.join(res_dir, 'csv_output.csv'))


if __name__ == '__main__':
    csv = r'C:\work_space\temp\k.csv'
    output_dir = r'C:\work_space\temp\t3'
    analysis(csv, output_dir)