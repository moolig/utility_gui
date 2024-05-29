import cv2
import os
import PySimpleGUI as sg
from os.path import join

def do_for_all_im_in_folder(input_path: str, output_path: str, function):
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    im_files = os.listdir(input_path)
    file_names = [im_file.split('_')[0] for im_file in im_files if '_Image.png' in im_file]
    for file_name in file_names:
        function(input_path, output_path, file_name)

def remove_mask_from_cad(root: str, output_path: str, base_name: str):
    # Read CAD image and mask
    cad_img = cv2.imread(join(root, f'{base_name}_Cad.png'), cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(join(output_path, f'{base_name}_oldCad.png'), cad_img)
    mask_img = cv2.imread(join(root, f'{base_name}_Mask.png'), cv2.IMREAD_GRAYSCALE)

    # Ensure the images have the same size
    mask_img = cv2.resize(mask_img, (cad_img.shape[1], cad_img.shape[0]))

    # Convert mask to binary (0 or 255)
    _, mask_binary = cv2.threshold(mask_img, 1, 255, cv2.THRESH_BINARY)

    # Invert the mask (0 for regions to remove, 255 for regions to keep)
    mask_inverted = cv2.bitwise_not(mask_binary)

    # Set the alpha channel of CAD image to 0 for masked regions
    new_cad_img = cv2.bitwise_and(cad_img, mask_inverted)

    # Save the resulting image

    cv2.imwrite(join(output_path, f'{base_name}_Cad.png'), new_cad_img)



data_folder = [
    [
        sg.Text("Image Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-DATA FOLDER-"),
        sg.FolderBrowse(),
    ],
]


layout = [
    [
        [sg.Column(data_folder)],
        [sg.Button("RUN remove mask from cad")]
    ]
]

# Create the window
window = sg.Window("remove mask from cad", layout)

# Create an event loop
while True:
    event, values = window.read()

    if event == "-DATA FOLDER-":
        data_FOLDER = values["-DATA FOLDER-"]


    if event == "RUN remove mask from cad":
        do_for_all_im_in_folder(data_FOLDER,
                                data_FOLDER, remove_mask_from_cad)

    if event == sg.WIN_CLOSED:
        break

window.close()