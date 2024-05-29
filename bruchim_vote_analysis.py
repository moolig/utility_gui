import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def func(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%\n({:d})".format(pct, absolute)
def create_graph(df, not_relevant_columns: list, save_dir: str):
    for column in df.columns:
        if column not in not_relevant_columns:
            value_counts = df[column].value_counts()
            y = np.array(value_counts.values)
            mylabels = [idx[::-1] for idx in value_counts.index]

            plt.pie(y, labels=mylabels, autopct = lambda pct: func(pct, y))
            plt.savefig(os.path.join(save_dir, column + '.png'))
            plt.show()
            plt.close()


def create_column_chart(df, not_relevant_columns: list, save_dir: str):
    for column in df.columns:
        if column not in not_relevant_columns:
            names_list = df[column].str.split(', ')
            all_names = [name.strip() for sublist in names_list for name in sublist]
            # revers name for convert Hebrew names
            all_names = [idx[::-1] for idx in all_names]

            name_counts = pd.Series(all_names).value_counts()
            plt.figure(figsize=(10, 8))
            name_counts.plot(kind='bar', color='skyblue')
            plt.title('Count of Names Selected')
            plt.xlabel('Names')
            plt.ylabel('Count')
            plt.xticks(rotation=90)
            plt.tight_layout()
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

def analysis(csv_input: str, res_dir: str, graph_type='pie'):
    pin_filed = 'סיסמה'
    point_filed = 'Score'#'ניקוד'
    good_val = '100 / 100'
    not_relevant_column = [pin_filed, point_filed, 'Timestamp']#'חותמת זמן']

    df = pd.read_csv(csv_input, sep=",", encoding='utf-8')
    print('incurrect pins:', get_incorrect_pins(df, pin_filed, point_filed, good_val))
    print('duplicate pin:', get_duplicate_pin(df, pin_filed))

    with open(os.path.join(res_dir, 'incorrect_pins.txt'), 'w') as the_file:
        the_file.write(f'incorrect pins: {get_incorrect_pins(df, pin_filed, point_filed, good_val)}\n')
        the_file.write(f'duplicate pin:{get_duplicate_pin(df, pin_filed)}\n')

    df = remove_incoret_pin(df, point_filed, good_val)
    df = fusion_same_pin(df, pin_filed)
    if(graph_type == 'pie'):
        create_graph(df, not_relevant_column, res_dir)
    elif graph_type == 'bar':
        create_column_chart(df, not_relevant_column, res_dir)
    df.to_csv(os.path.join(res_dir, 'csv_output.csv'))

# def analysis_from_google_sheet(url, res_dir):
#     df = pd.read_csv(url, on_bad_lines='skip')
#     print('incurrect pins:', get_incorrect_pins(df))
#     print('duplicate pin:', get_duplicate_pin(df))
#     df = remove_incoret_pin(df)
#     df = fusion_same_pin(df)
#     df.to_csv(os.path.join(res_dir, 'csv_output.csv'))


# if __name__ == '__main__':
#     csv = r'C:\work_space\temp\kamin_2\input.csv'
#     output_dir = r'C:\work_space\temp\kamin_2'
#     analysis(csv, output_dir, graph_type='bar')





import os
import PySimpleGUI as sg


input_csv = [
    [
        sg.Text("select input csv file"),
        sg.In(size=(25, 1), enable_events=True, key="-input_file-"),
        sg.FileBrowse(),
    ],
]

output_folder = [
    [
        sg.Text("output path"),
        sg.In(size=(25, 1), enable_events=True, key="-output_path-"),
        sg.FolderBrowse(),
    ],
]

graph_type_select = [
    [
        sg.Text("graph type"),
        # sg.In(size=(25, 1), enable_events=True, key="-graph_type-"),
        sg.Listbox(["pie", "bar"], size=(10, 2), key="-graph_type-"),
    ],
]


layout = [
    [
        [sg.Column(input_csv)],
        [sg.Column(output_folder)],
        [sg.Column(graph_type_select)],
        [sg.Button("RUN")]
    ]
]


# Create the window
window = sg.Window("vote analysis", layout)


# Create an event loop
while True:
    event, values = window.read()


    if event == "-input_file-":
        input_file = values["-input_file-"]
    if event == "-output_path-":
        output_path = values["-output_path-"]

    graph_type = values["-graph_type-"][0]

    if event == "RUN":
        analysis(input_file, output_path, graph_type=graph_type)
    if event == sg.WIN_CLOSED:
        break

window.close()