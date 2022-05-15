import shutil
import os


def recup_img(directorie_inpute, directorie_outpute32, directorie_outpute_origin):
    # code to move the files from sub-folder to main folder.
    files = os.listdir(directorie_inpute)

    path = os.getcwd()
    path = os.path.join(path, directorie_outpute32)
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.getcwd()
    path = os.path.join(path, directorie_outpute_origin)
    if not os.path.exists(path):
        os.mkdir(path)

    for file in files:
        file_name = os.path.join(directorie_inpute, file)
        if "modif" not in file_name:
            shutil.copy(file_name, directorie_outpute_origin)
        else:
            shutil.copy(file_name, directorie_outpute32)

    print("Files Moved")
