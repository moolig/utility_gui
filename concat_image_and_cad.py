import cv2
import os
import numpy as np
import PySimpleGUI as sg



def channel_concat_image_and_cad(input_im_dir: str, input_cad_dir: str,  output_dir):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    im_files = os.listdir(input_im_dir)

    file_names = [im_file.split('_')[0] for im_file in im_files if '.png' in im_file]
    for file_name in file_names:
        image = cv2.imread(os.path.join(input_im_dir, f'{file_name}_Image.png'), cv2.IMREAD_GRAYSCALE)
        cad = cv2.imread(os.path.join(input_cad_dir, f'{file_name}_Cad.png'), cv2.IMREAD_GRAYSCALE)

        image_merged = cv2.merge((image, image, (cad/2).astype('uint8')))
        # image_merged = cv2.merge((image, image, cad))

        cv2.imwrite(os.path.join(output_dir, f'{file_name}_merge.png'), image_merged)



def channel_concat_rgb_image_and_cad(input_im_dir: str, input_cad_dir: str,  output_dir):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    im_files = os.listdir(input_im_dir)

    file_names = [im_file.split('_')[0] for im_file in im_files if '.png' in im_file]
    for file_name in file_names:
        image = cv2.imread(os.path.join(input_im_dir, f'{file_name}_Image.png'), cv2.IMREAD_COLOR)
        cad = cv2.imread(os.path.join(input_cad_dir, f'{file_name}_Cad.png'), cv2.IMREAD_GRAYSCALE)

        image_merged = cv2.merge((image, (cad/2).astype('uint8')))
        # image_merged = cv2.merge((image, image, cad))

        cv2.imwrite(os.path.join(output_dir, f'{file_name}_merge.png'), image_merged)



def concat_image_and_cad(input_im_dir: str, input_cad_dir: str,  output_dir):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    im_files = os.listdir(input_im_dir)

    file_names = [im_file.split('_')[0] for im_file in im_files if '.png' in im_file]
    for file_name in file_names:
        image = cv2.imread(os.path.join(input_im_dir, f'{file_name}_Image.png'))
        cad = cv2.imread(os.path.join(input_cad_dir, f'{file_name}_Cad.png'))
        concatenated = np.concatenate((image, cad), axis=1)
        cv2.imwrite(os.path.join(output_dir, f'{file_name}_concat.png'), concatenated)


im_folder = [
    [
        sg.Text("Image Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-im FOLDER-"),
        sg.FolderBrowse(),
    ],
]

cad_folder = [
    [
        sg.Text("Cad Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-cad FOLDER-"),
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


layout = [
    [
        [sg.Column(im_folder)],
        [sg.Column(cad_folder)],
        [sg.Column(output_folder)],
        [sg.Button("RUN channel concat"), sg.Button("RUN concat"), sg.Button("RUN RGB channel concat")]
    ]
]

# Create the window
window = sg.Window("channel_concat_image_and_cad", layout)

# Create an event loop
while True:
    event, values = window.read()


    if event == "-im FOLDER-":
        im_FOLDER = values["-im FOLDER-"]
    if event == "-cad FOLDER-":
        cad_FOLDER = values["-cad FOLDER-"]
    if event == "-res FOLDER-":
        res_FOLDER = values["-res FOLDER-"]


    if event == "RUN channel concat":
        channel_concat_image_and_cad(im_FOLDER, cad_FOLDER, res_FOLDER)

    if event == "RUN RGB channel concat":
        channel_concat_rgb_image_and_cad(im_FOLDER, cad_FOLDER, res_FOLDER)

    if event == "RUN concat":
        concat_image_and_cad(im_FOLDER, cad_FOLDER, res_FOLDER)
    if event == sg.WIN_CLOSED:
        break

window.close()