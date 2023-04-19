import os
import PySimpleGUI as sg
import sys
import cv2
sys.path.insert(0, r"C:/AI/utility")

from annotations import Annotations



def resize_annotations_in_dir(dir_in: str, dir_out, old_im_size, new_image_size):
    '''
    this function is relevant only if we add to image in right or in bottom
    or if we remove from right or from bottom and no roi in remove part
    '''
    if not os.path.isdir(dir_out):
        os.mkdir(dir_out)
    files = [f for f in os.listdir(dir_in)]
    for file in files:
        annotations = Annotations(yolo_file=os.path.join(dir_in, file))
        annotations.resize_im(old_im_size, new_image_size)
        annotations.save_to_yolo_format(os.path.join(dir_out, file))


in_folder = [
    [
        sg.Text("input Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-in FOLDER-"),
        sg.FolderBrowse(),
    ],
]

output_folder = [
    [
        sg.Text("Output Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-res FOLDER-"),
        sg.FolderBrowse(),
    ],
]

orig_size = [
    [
        sg.Text("im orig size"),
        sg.Combo([(256, 128), (128, 256), (128, 128), (256, 256)], default_value=(256, 128), key="-orig_size-")
    ]
]

new_size = [
    [
        sg.Text("new image size"),
        sg.Combo([(256, 128), (128, 256), (128, 128), (256, 256)], default_value=(256, 128), key="-new_size-")
    ]
]

layout = [
    [
        [sg.Column(in_folder)],
        [sg.Column(output_folder)],
        [sg.Column(orig_size), sg.Column(new_size)],
        [sg.Button("RUN")]

    ]
]

window = sg.Window("rename_file_in_directory", layout)

# Create an event loop
while True:
    event, values = window.read()

    if event == "-in FOLDER-":
        dir_in = values["-in FOLDER-"]
    if event == "-res FOLDER-":
        dir_out = values["-res FOLDER-"]

    if event == "RUN":
        im_orig_size = values['-orig_size-']
        new_im_size = values['-new_size-']
        resize_annotations_in_dir(dir_in, dir_out, im_orig_size, new_im_size)
    if event == sg.WIN_CLOSED:
        break

window.close()