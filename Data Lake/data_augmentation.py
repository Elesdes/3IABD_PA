import glob
import random
from PIL import Image
from imgaug import augmenters as iaa
import cv2


images = []
image_path = glob.glob("EiffelTower_picture/Pinterest/*.jpg")

for img_path in image_path:
    img = cv2.imread(img_path)
    images.append(img)

def write_img(path, transform, ibis):
    pictures = transform(images=images)
    i = 1
    for img in pictures:
        url = path + str(i) + "." + str(ibis) + ".jpg"
        i+=1
        # cv2.imshow("Image", img)
        cv2.imwrite(url, img)
        cv2.waitKey(0)

nb_img = len(images)

# crop de 1% a 10% de l'image
def Crop_picture_aug1():
    randCrop = random.uniform(0.01,0.3)
    Crop_picture_aug = iaa.Sequential([
        # iaa.Affine(translate_percent={"x": (0.2)}),
        iaa.imgaug.augmenters.size.Crop(percent=randCrop)
    ])
    return Crop_picture_aug

def Crop_picture_aug2():
    randCrop = random.uniform(0.01,0.3)
    Crop_picture_aug = iaa.Sequential([
        # iaa.Affine(translate_percent={"x": (0.2)}),
        iaa.imgaug.augmenters.size.Crop(percent=randCrop)
    ])
    return Crop_picture_aug

ibis = 1
path = 'EiffelTower_picture/Pinterest_modify/crop/img'
transforme = Crop_picture_aug1()
write_img(path, transforme, ibis)
ibis += 1
transforme = Crop_picture_aug2()
write_img(path, transforme, ibis)

# applique un effet de miroire a l'image 100% du temps
def flip_picture_aug():
    flip_picture_aug = iaa.Sequential([
        iaa.Fliplr(1)
    ])
    return flip_picture_aug

ibis = 1
path = 'EiffelTower_picture/Pinterest_modify/flip/img'
transforme = flip_picture_aug()
write_img(path, transforme, ibis)

# floute l'image
def Blur_picture_aug():
    Blur_picture_aug = iaa.Sequential([
        iaa.GaussianBlur(sigma=(1, 2.5))
    ])
    return Blur_picture_aug

ibis = 1
path = 'EiffelTower_picture/Pinterest_modify/blur/img'
for j in range(3):
    transform = Blur_picture_aug()
    write_img(path, transform, ibis)
    ibis += 1

# zoom et dezoom l'image
def zoom_picture_aug():
    zoom_picture_aug = iaa.Sequential([
        iaa.Affine(scale= (0.8,1.2))
    ])
    return zoom_picture_aug

ibis = 1
path = 'EiffelTower_picture/Pinterest_modify/zoom/img'
for j in range(3):
    transform = zoom_picture_aug()
    write_img(path, transform, ibis)
    ibis += 1

# modifie le contrastre de l'image en pos
def contrast_neg_picture_aug():
    contrast_neg_picture_aug = iaa.Sequential([
        iaa.GammaContrast((0.3, 1))
    ])
    return contrast_neg_picture_aug

# modifie le contrastre de l'image en neg
def contrast_pos_picture_aug():
    contrast_pos_picture_aug = iaa.Sequential([
        iaa.GammaContrast((1, 2))
    ])
    return contrast_pos_picture_aug

ibis = 1
path = 'EiffelTower_picture/Pinterest_modify/contrast/img'
transforme = contrast_neg_picture_aug()
write_img(path, transforme, ibis)
ibis += 1
transforme = contrast_pos_picture_aug()
write_img(path, transforme, ibis)

# modifie la saturation de l'image
def Saturation_picture_aug():
    Saturation_picture_aug = iaa.Sequential([
        iaa.RemoveSaturation((0.1,0.8))
    ])
    return Saturation_picture_aug

ibis = 1
path = 'EiffelTower_picture/Pinterest_modify/saturation/img'
for j in range(3):
    transform = Saturation_picture_aug()
    write_img(path, transform, ibis)
    ibis += 1

# modifie la temperature de l'image
def Temperature_picture_aug():
    Temperature_picture_aug = iaa.Sequential([
        iaa.ChangeColorTemperature([3000,5000,10000,20000,30000,40000])
    ])
    return Temperature_picture_aug

ibis = 1
path = 'EiffelTower_picture/Pinterest_modify/temperature/img'
for j in range(3):
    transform = Temperature_picture_aug()
    write_img(path, transform, ibis)
    ibis += 1

# modifie luminosité de l'image en pos et neg
def Brightness_picture_aug():
    Brightness_picture_aug = iaa.Sequential([
        iaa.BlendAlpha((0.0, 1.0),foreground=iaa.Add(100),background=iaa.Multiply(0.2))
    ])
    return Brightness_picture_aug

ibis = 1
path = 'EiffelTower_picture/Pinterest_modify/brightness/img'
for j in range(3):
    transform = Brightness_picture_aug()
    write_img(path, transform, ibis)
    ibis += 1

# modifie perspective de l'image
def Wrappe_picture_aug():
    Wrappe_picture_aug = iaa.Sequential([
        iaa.PiecewiseAffine(scale=(0.01,0.05))
    ])
    return Wrappe_picture_aug

ibis = 1
path = 'EiffelTower_picture/Pinterest_modify/wrappe/img'
for j in range(3):
    transform = Wrappe_picture_aug()
    write_img(path, transform, ibis)
    ibis += 1

# reotation l'image en pos et neg
def Rotate_picture_aug():
    Rotate_picture_aug = iaa.Sequential([
        iaa.pillike.Affine(rotate=(-45, 45), fillcolor=(0, 256))
    ])
    return Rotate_picture_aug

ibis = 1
path = 'EiffelTower_picture/Pinterest_modify/rotate/img'
for j in range(3):
    transform = Rotate_picture_aug()
    write_img(path, transform, ibis)
    ibis += 1

# simule soleil
def Sun_picture_aug():
    Sun_picture_aug = iaa.Sequential([
        iaa.BlendAlphaMask(
            iaa.InvertMaskGen(1,iaa.VerticalLinearGradientMaskGen(0.2, 1, 0, 1)),
            iaa.Sequential([
                iaa.Clouds(),
                iaa.WithChannels((1, 2), iaa.Multiply(3))
            ])
        ),
    
        iaa.BlendAlphaMask(
            iaa.InvertMaskGen(0.5, iaa.HorizontalLinearGradientMaskGen(0.2, 1, 0, 1)),
            iaa.Sequential([
                iaa.Clouds(),
                iaa.WithChannels((1, 2), iaa.Multiply(3))
            ])
        )
    ])
    return Sun_picture_aug

ibis = 1
path = 'EiffelTower_picture/Pinterest_modify/sun/img'
for j in range(3):
    transform = Sun_picture_aug()
    write_img(path, transform, ibis)
    ibis += 1

# simule pluie
def rain_picture_aug():
    rain_picture_aug = iaa.Sequential([
        iaa.Rain(speed=(0.1, 0.5))
    ])
    return rain_picture_aug

ibis = 1
path = 'EiffelTower_picture/Pinterest_modify/rain/img'
for j in range(3):
    transform = rain_picture_aug()
    write_img(path, transform, ibis)
    ibis += 1

#qualité l'image en pos et neg
def quality_picture_aug():
    quality_picture_aug23 = iaa.Sequential([
        iaa.BlendAlpha([0.25, 0.75], iaa.MedianBlur(13)),
        iaa.JpegCompression(compression=(50, 90))
    ])
    return quality_picture_aug23

ibis = 1
path = 'EiffelTower_picture/Pinterest_modify/quality/img'
for j in range(3):
    transform = quality_picture_aug()
    write_img(path, transform, ibis)
    ibis += 1

#bruit l'image en pos et neg
def Noise_picture_aug():
    Noise_picture_aug = iaa.Sequential([
        iaa.AdditiveGaussianNoise(loc=0, scale=(30,50), per_channel=0.5)
    ])
    return Noise_picture_aug

ibis = 1
path = 'EiffelTower_picture/Pinterest_modify/noise/img'
for j in range(3):
    transform = Noise_picture_aug()
    write_img(path, transform, ibis)
    ibis += 1

# met des carré/rectange de couleur aléatoirement sur l'image
def Cutout_colores_picture_aug():
    Cutout_colores_picture_aug = iaa.Sequential([
        iaa.Cutout(fill_mode="constant",nb_iterations=(1, 5), cval=(0, 255),squared=False, fill_per_channel=0.5)
    ])
    return Cutout_colores_picture_aug

ibis = 1
path = 'EiffelTower_picture/Pinterest_modify/cutout_colores/img'
for j in range(3):
    transform = Cutout_colores_picture_aug()
    write_img(path, transform, ibis)
    ibis += 1

# met des carré/rectange en bruit aléatoirement sur l'image
def Cutout_noise_picture_aug():
    Cutout_noise_picture_aug = iaa.Sequential([
        iaa.Cutout(fill_mode="gaussian", nb_iterations=(1, 5),squared=False, fill_per_channel=True)
    ])
    return Cutout_noise_picture_aug

ibis = 1
path = 'EiffelTower_picture/Pinterest_modify/cutout_noise/img'
for j in range(3):
    transform = Cutout_noise_picture_aug()
    write_img(path, transform, ibis)
    ibis += 1

# met des petits carrés de couleur transparent aléatoirement sur l'image
def Coarse_Dropout_picture_aug():
    Coarse_Dropout_picture_aug = iaa.Sequential([
        iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5)
    ])
    return Coarse_Dropout_picture_aug

ibis = 1
path = 'EiffelTower_picture/Pinterest_modify/coarse_dropout/img'
for j in range(3):
    transform = Coarse_Dropout_picture_aug()
    write_img(path, transform, ibis)
    ibis += 1

def nothing():
    nothing = iaa.Sequential([
    ])
    return nothing

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
        proba(zoom_picture_aug(), nothing),
        proba(Brightness_picture_aug(), nothing),
        proba(p, nothing),
        proba(Wrappe_picture_aug(), nothing)
    ])
    return get_transforms



ibis = 0
path = 'EiffelTower_picture/Pinterest_modify/get_transforms/img'
for j in range(3):
    transform = get_transforms()
    write_img(path, transform, ibis)
    ibis += 1

def modif_luminosité():
    imge = "EiffelTower_picture/Pinterest/img4.jpg"
    largeur_image = 236
    hauteur_image = 354
    for p in range(20):
        img = Image.open(imge)
        rand = random.randrange(-150,150)
        print(rand)
        for y in range(hauteur_image):
            for x in range(largeur_image):
                r, v, b = img.getpixel((x, y))
                n_r = r - rand
                n_v = v - rand
                n_b = b - rand
                img.putpixel((x, y), (n_r, n_v, n_b))
        img.show()