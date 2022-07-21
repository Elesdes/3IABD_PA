import glob
import random
from data_augmentation import *
from imgaug import augmenters as iaa
import cv2
from tqdm import tqdm

def modifications_size(directorie_inpute,directorie_outpute,size):
    len_dir = len(directorie_inpute)
    directorie_inpute = directorie_inpute + "/*.jpg"
    image_path = glob.glob(directorie_inpute)

    def write_img(path, transform,name,images):
        pictures = transform(images=images)
        i = 0
        for img in (pictures):
            url = path + name[i][:-4] + "_modif" + name[i][-4:]
            i += 1
            cv2.imwrite(url, img)
            cv2.waitKey(0)

    # applique un effet de miroire a l'image 100% du temps
    def flip_picture_aug(size):
        flip_picture_aug = iaa.Resize({"height": size, "width": size})
        return flip_picture_aug

    for img_path in tqdm(image_path):
        name = []
        images = []
        img = cv2.imread(img_path)
        name.append(img_path[len_dir:])
        images.append(img)
        path = directorie_outpute
        transforme = flip_picture_aug(size)
        write_img(path, transforme,name,images)

def modifications_color_BW(directorie_inpute,directorie_outpute):
    len_dir = len(directorie_inpute)
    directorie_inpute = directorie_inpute + "/*.jpg"
    image_path = glob.glob(directorie_inpute)


    def write_img(path, transform, name, images):
        pictures = transform(images=images)
        i = 0
        for img in (pictures):
            url = path + name[i][:-10] + "_BX" + name[i][-4:]
            i += 1
            cv2.imwrite(url, img)
            cv2.waitKey(0)

    # applique un effet de miroire a l'image 100% du temps
    def flip_picture_aug():
        flip_picture_aug = iaa.Grayscale(alpha=1.0)
        return flip_picture_aug

    for img_path in tqdm(image_path):
        name = []
        images = []
        img = cv2.imread(img_path)
        name.append(img_path[len_dir:])
        images.append(img)
        path = directorie_outpute
        transforme = flip_picture_aug()
        write_img(path, transforme,name, images)


def aug(directorie_inpute,directorie_outpute, prefix):
    len_dir = len(directorie_inpute)
    directorie_inpute = directorie_inpute + "/*.jpg"
    image_path = glob.glob(directorie_inpute)

    def write_img(path, transform, name, images):
        pictures = transform(images=images)
        i = 0
        for img in (pictures):
            url = path + name[i][:-4] + prefix + name[i][-4:]
            i += 1
            cv2.imwrite(url, img)
            cv2.waitKey(0)

    def get_transforms():
        def proba(modif, nothing):
            rand_percent = random.randrange(0, 100)
            if rand_percent < 75:
                i = modif
            else:
                i = nothing()
            return i

        rand_contraste = random.uniform(0, 1)
        if rand_contraste:
            p = contrast_neg_picture_aug()
        else:
            p = contrast_pos_picture_aug()
        get_transforms = iaa.Sequential([
            proba(Crop_picture_aug1(), nothing),
            proba(flip_picture_aug(), nothing),
            proba(Brightness_picture_aug(), nothing),
            proba(p, nothing),
            proba(Wrappe_picture_aug(), nothing)
        ])
        return get_transforms

    for img_path in tqdm(image_path):
        name = []
        images = []
        img = cv2.imread(img_path)
        name.append(img_path[len_dir:])
        images.append(img)
        path = directorie_outpute
        transforme = get_transforms()
        write_img(path, transforme, name, images)