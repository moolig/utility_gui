import os
import PySimpleGUI as sg

def remove_not_in_both_folder(dirA: str, dirB: str):
    '''
    remove from dirB if not in dirA
    '''
    onlyfiles = [f for f in os.listdir(dirA) if os.path.isfile(os.path.join(dirA, f))]
    onlyfiles = [f.split('.')[0] for f in onlyfiles]

    onlyim = [f for f in os.listdir(dirB) if os.path.isfile(os.path.join(dirB, f))]
    for im in onlyim:
        im_name = im.split('.')[0]
        if im_name not in onlyfiles:
            os.unlink(os.path.join(dirB, im))


file_list_column_A = [
    [
        sg.Text("Reference Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-Reference FOLDER-"),
        sg.FolderBrowse(),
    ],
]

file_list_column_B = [
    [
        sg.Text("Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ],
]


layout = [
    [
        [sg.Column(file_list_column_A)],
        [sg.Column(file_list_column_B)],
        [sg.Button("RUN")]
    ]
]

# Create the window
window = sg.Window("remove_not_in_both_folder", layout)


# Create an event loop
while True:
    event, values = window.read()


    if event == "-Reference FOLDER-":
        Reference_FOLDER = values["-Reference FOLDER-"]
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]


    if event == "RUN":
        remove_not_in_both_folder(Reference_FOLDER, folder)
    if event == sg.WIN_CLOSED:
        break

window.close()
