# for image in all_anchors2:
#     if (image.find('jpg') != -1) or (image.find('jpeg') != -1):
#         ext = ".jpg"
#     elif (image.find('png') != -1):
#         ext = ".png"
#
#     save_as = os.path.join(path, "img" + str(counter) + ext)
#     # print(image, '\n' ,save_as)
#     wget.download(image, save_as)
#     counter += 1
#
#
import os
import wget
def download(directorie_outpute, directorie_link):
    counter = 0
    path = os.getcwd()
    path = os.path.join(path, directorie_outpute)
    with open(directorie_link, "r") as filin:
        for ligne in filin:
            print(ligne)
            if (ligne.find('jpg') != -1) or (ligne.find('jpeg') != -1):
                    ext = ".jpg"
            elif (ligne.find('png') != -1):
                ext = ".png"

            save_as = os.path.join(path, "img" + str(counter) + ext)
            wget.download(ligne, save_as)
            counter += 1