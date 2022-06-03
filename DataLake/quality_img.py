import glob
from imgaug import augmenters as iaa
import cv2
from tqdm import tqdm

def modifications_size(directorie_inpute,directorie_outpute,size):
    images = []
    len_dir = len(directorie_inpute)
    directorie_inpute = directorie_inpute + "/*.jpg"
    image_path = glob.glob(directorie_inpute)
    name = []
    for img_path in tqdm(image_path):
        img = cv2.imread(img_path)
        name.append(img_path[len_dir:])
        images.append(img)

    def write_img(path, transform,name):
        pictures = transform(images=images)
        i = 0
        for img in tqdm(pictures):
            url = path + name[i][:-4] + "_modif" + name[i][-4:]
            i += 1
            cv2.imwrite(url, img)
            cv2.waitKey(0)

    # applique un effet de miroire a l'image 100% du temps
    def flip_picture_aug(size):
        flip_picture_aug = iaa.Resize({"height": size, "width": size})
        return flip_picture_aug

    path = directorie_outpute
    transforme = flip_picture_aug(size)
    write_img(path, transforme,name)

def modifications_color_BW(directorie_inpute,directorie_outpute):
    images = []
    len_dir = len(directorie_inpute)
    directorie_inpute = directorie_inpute + "/*.jpg"
    image_path = glob.glob(directorie_inpute)
    name = []
    for img_path in tqdm(image_path):
        img = cv2.imread(img_path)
        name.append(img_path[len_dir:])
        images.append(img)

    def write_img(path, transform,name):
        pictures = transform(images=images)
        i = 0
        for img in tqdm(pictures):
            url = path + name[i][:-10] + "_BX" + name[i][-4:]
            i += 1
            cv2.imwrite(url, img)
            cv2.waitKey(0)

    # applique un effet de miroire a l'image 100% du temps
    def flip_picture_aug():
        flip_picture_aug = iaa.Grayscale(alpha=1.0)
        return flip_picture_aug

    path = directorie_outpute
    transforme = flip_picture_aug()
    write_img(path, transforme,name)