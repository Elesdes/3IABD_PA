import random
import shutil
import os


def test_train(directorie_inpute, directorie_outputetest, directorie_outpute_train):
    # code to move the files from sub-folder to main folder.
    files = os.listdir(directorie_inpute)

    path = os.getcwd()
    path = os.path.join(path, directorie_outputetest)
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.getcwd()
    path = os.path.join(path, directorie_outpute_train)
    if not os.path.exists(path):
        os.mkdir(path)

    for file in files:
        rand = random.randint(0, 100)
        file_name = os.path.join(directorie_inpute, file)
        if rand <= 80:
            shutil.copy(file_name, directorie_outpute_train)
        else:
            shutil.copy(file_name, directorie_outputetest)

    print("Files Moved")
