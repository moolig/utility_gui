import os
import PySimpleGUI as sg

def rename_file_in_directory(directory_path, old_sub_file_name, new_sub_file_name):
  # Get list of files in directory
  files = [f for f in os.listdir(directory_path)]


  # Iterate over all the files in the directory
  for file in files:
    # Create new file name
    new_name = file.replace(old_sub_file_name, new_sub_file_name)
    # Rename the file
    try:
        os.rename(os.path.join(directory_path, file), os.path.join(directory_path, new_name))
    except Exception:
        print(file)
		


file_list_column = [
    [
        sg.Text("Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ],
]

old_sub_file_name = [
    [
        sg.Text("old sub file name"),
        sg.Input(size=(25, 1), enable_events=True, key="-old_sub_file_name-")
    ]
]

new_sub_file_name = [
    [
        sg.Text("old sub file name"),
        sg.Input(size=(25, 1), enable_events=True, key="-new_sub_file_name-")
    ]
]


layout = [
    [
        [sg.Column(file_list_column)],
        [sg.Column(old_sub_file_name)],
        [sg.Column(new_sub_file_name)],
        [sg.Button("RUN")]

    ]
]

# Create the window
window = sg.Window("rename_file_in_directory", layout)
old_sub_file_name_str = ''
new_sub_file_name_str = ''

# Create an event loop
while True:
    event, values = window.read()

    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
    # if event == "-old_sub_file_name-":
    #     old_sub_file_name_str += values['-old_sub_file_name-'][-1]
    # if event == "-new_sub_file_name-":
    #     new_sub_file_name_str += values['-new_sub_file_name-'][-1]

    if event == "RUN":
        old_sub_file_name_str += values['-old_sub_file_name-']
        new_sub_file_name_str += values['-new_sub_file_name-']

        rename_file_in_directory(folder, old_sub_file_name_str, new_sub_file_name_str)
    if event == sg.WIN_CLOSED:
        break

window.close()
