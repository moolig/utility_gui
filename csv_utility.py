import pandas as pd

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
    grup_scores_sheet_name = 'grup scores'

    df = pd.read_csv(csv_input, sep=",", encoding='cp1252')
    df['name'] = df['name'].str.lower().str.strip()
    avg_scores = df.groupby('name')['points'].mean()
    res_frame = avg_scores.to_frame()
    labeling_count = df.groupby('name')['points'].count()
    res_frame['labeling #'] = labeling_count

    res_frame.to_csv(csv_output)



if __name__ == '__main__':
    # excel_path = r'\\10.3.3.13\pcb\AI\DataSets\scores_table.xlsx'
    # name_score_from_excel(excel_path)
    name_score_from_csv(r"\\10.3.3.13\pcb\AI\DataSets\grup_scores.csv", r'\\10.3.3.13\pcb\AI\DataSets\res_grup_scores.csv')