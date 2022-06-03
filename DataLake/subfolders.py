import shutil
import os
def divided_into_subfolders(directorie_inpute):
    # Define the source and destination path
    destination1 = directorie_inpute + "/sub1/"
    destination2 = directorie_inpute + "/sub2/"
    destination3 = directorie_inpute + "/sub3/"


    # code to move the files from sub-folder to main folder.
    files = os.listdir(directorie_inpute)
    i=0

    dest1 = list(range(1, len(files) + 1, 6))
    dest1.extend(list(range(2, len(files) + 1, 6)))
    dest2 = list(range(3, len(files) + 1, 6))
    dest2.extend(list(range(4, len(files) + 1, 6)))
    dest3 = list(range(5, len(files) + 1, 6))
    dest3.extend(list(range(6, len(files) + 1, 6)))

    path = os.getcwd()
    path = os.path.join(path, destination1)
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.getcwd()
    path = os.path.join(path, destination2)
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.getcwd()
    path = os.path.join(path, destination3)
    if not os.path.exists(path):
        os.mkdir(path)

    for file in files:
        i+=1
        if i in dest1:
            file_name = os.path.join(directorie_inpute, file)
            shutil.move(file_name, destination1)
        elif i in dest2:
            file_name = os.path.join(directorie_inpute, file)
            shutil.move(file_name, destination2)
        elif i in dest3:
            file_name = os.path.join(directorie_inpute, file)
            shutil.move(file_name, destination3)
    print("Files Moved")