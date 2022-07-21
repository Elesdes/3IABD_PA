import os
import shutil
from os import listdir, rename
from os.path import isfile, join
import tkinter as tk
from PIL import ImageTk, Image
from tqdm import tqdm


def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()

def tri_bulle(tab):
    n = len(tab)
    # Traverser tous les éléments du tableau
    for i in tqdm(range(n)):
        for j in range(0, n-i-1):
            # échanger si l'élément trouvé est plus grand que le suivant
            if is_integer(tab[j][3:-4]) and is_integer(tab[j+1][3:-4]):
                if int(tab[j][3:-4]) > int(tab[j+1][3:-4]):
                    tab[j], tab[j+1] = tab[j+1], tab[j]
            elif not is_integer(tab[j][3:-4]) and not is_integer(tab[j+1][3:-4]):
                if int(tab[j][3:-10]) > int(tab[j + 1][3:-10]):
                    tab[j], tab[j+1] = tab[j+1], tab[j]

            elif is_integer(tab[j][3:-4]) and not is_integer(tab[j+1][3:-4]):
                if int(tab[j][3:-4]) > int(tab[j + 1][3:-10]):
                    tab[j], tab[j+1] = tab[j+1], tab[j]

            elif not is_integer(tab[j][3:-4]) and is_integer(tab[j+1][3:-4]):
                if int(tab[j][3:-10]) > int(tab[j + 1][3:-4]):
                    tab[j], tab[j+1] = tab[j+1], tab[j]

    return tab

# lien avec vos img non trie
IMAGE_FOLDER = "D:/PA/pantheon_picture/all_clean/sub3"

images = [f for f in listdir(IMAGE_FOLDER) if isfile(join(IMAGE_FOLDER, f))]
images = tri_bulle(images)
print(images)
unclassified_images = filter(lambda image: not (image.startswith("0_") or image.startswith("1_")), images)
current = None

def next_img():
    global current, unclassified_images
    try:
        current = next(unclassified_images)
    except StopIteration:
        root.quit()
    print(current)
    pil_img = Image.open(IMAGE_FOLDER+"/"+current)
    width, height = pil_img.size
    max_height = 1000
    if height > max_height:
        resize_factor = max_height / height
        pil_img = pil_img.resize((int(width*resize_factor), int(height*resize_factor)), resample=Image.LANCZOS)
    img_tk = ImageTk.PhotoImage(pil_img)
    img_label.img = img_tk
    img_label.config(image=img_label.img)


def positive(arg):
    global current

    file_name = os.path.join(IMAGE_FOLDER, current)
    shutil.move(file_name, IMAGE_FOLDER + "/good")

    current = next(unclassified_images)

    file_name = os.path.join(IMAGE_FOLDER, current)
    shutil.move(file_name, IMAGE_FOLDER + "/good")

    next_img()

def negative(arg):
    global current

    file_name = os.path.join(IMAGE_FOLDER, current)
    shutil.move(file_name, IMAGE_FOLDER + "/bad")

    # rename(IMAGE_FOLDER + "/" + current, IMAGE_FOLDER + "/0_" + current)
    current = next(unclassified_images)

    file_name = os.path.join(IMAGE_FOLDER, current)
    shutil.move(file_name, IMAGE_FOLDER + "/bad")

    # rename(IMAGE_FOLDER + "/" + current, IMAGE_FOLDER + "/0_" + current)

    next_img()


if __name__ == "__main__":

    root = tk.Tk()
    img_label = tk.Label(root)
    img_label.pack()
    img_label.bind("<Button-1>", positive)
    img_label.bind("<Button-3>", negative)


    btn = tk.Button(root, text='Next image', command=next_img)

    next_img() # load first image

    root.mainloop()