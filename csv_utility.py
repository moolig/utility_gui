import pandas as pd
import PySimpleGUI as sg

def name_score_from_excel(excel_path: str):
    grup_scores_sheet_name = 'grup scores'
    df = pd.read_excel(io=excel_path, sheet_name=grup_scores_sheet_name)
    df['Name'] = df['Name'].str.lower().str.strip()
    avg_scores = df.groupby('Name')['point'].mean()

    res_sheet_name = 'persons scores'
    writer = pd.ExcelWriter(excel_path, engine='openpyxl')
    avg_scores.to_excel(writer, sheet_name=res_sheet_name)
    writer.close()

def name_score_from_csv(csv_input: str, csv_output: str):
    df = pd.read_csv(csv_input, sep=",", encoding='cp1252')
    df['name'] = df['name'].str.lower().str.strip()
    avg_scores = df.groupby('name')['points'].mean()
    res_frame = avg_scores.to_frame()
    labeling_count = df.groupby('name')['points'].count()
    res_frame['labeling #'] = labeling_count

    res_frame.to_csv(csv_output)


def remove_all_raw_from_b_if_in_a(a_file, b_file):
    # Read the Excel files into pandas dataframes
    a_df = pd.read_excel(a_file)
    b_df = pd.read_excel(b_file)

    # Identify the common rows between the two dataframes
    common_rows = a_df.merge(b_df, on=list(a_df.columns), how='inner').index

    # Remove the common rows from b_df
    b_df.drop(common_rows, inplace=True)

    # Save the updated b_df back to the Excel file
    b_df.to_excel(b_file, index=False)




file_a = [
    [
        sg.Text("file_a"),
        sg.In(size=(25, 1), enable_events=True, key="-file_a-"),
        sg.FileBrowse(),
    ],
]

file_b = [
    [
        sg.Text("file_b"),
        sg.In(size=(25, 1), enable_events=True, key="-file_b-"),
        sg.FileBrowse(),
    ],
]



layout = [
    [
        [sg.Column(file_a)],
        [sg.Column(file_b)],
        [sg.Button("RUN")]

    ]
]

# Create the window
window = sg.Window("remove row from b if in a", layout)


# Create an event loop
while True:
    event, values = window.read()

    if event == "-file_a-":
        file_a_obj = values["-file_a-"]
    if event == "-file_b-":
        file_b_obj = values["-file_b-"]

    if event == "RUN":
        remove_all_raw_from_b_if_in_a(file_a_obj, file_b_obj)
        break
    if event == sg.WIN_CLOSED:
        break

window.close()


# if __name__ == '__main__':
#     # excel_path = r'\\10.3.3.13\pcb\AI\DataSets\scores_table.xlsx'
#     # name_score_from_excel(excel_path)
#     remove_all_raw_from_b_if_in_a(r'C:\work_space\temp\a.xlsx', r'C:\work_space\temp\b.xlsx')
#     # name_score_from_csv(r"\\10.3.3.13\pcb\AI\DataSets\grup_scores.csv", r'\\10.3.3.13\pcb\AI\DataSets\res_grup_scores.csv')