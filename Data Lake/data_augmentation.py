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

# crop de 1% a 10% de l'image
randCrop = random.uniform(0.01,0.1)
Crop_picture_aug = iaa.Sequential([
    # iaa.Affine(translate_percent={"x": (0.2)}),
    iaa.imgaug.augmenters.size.Crop(percent=randCrop)
])

# applique un effet de miroire a l'image 100% du temps
flip_picture_aug = iaa.Sequential([
    iaa.Fliplr(1)
])

# floute l'image
blur_picture_aug = iaa.Sequential([
    iaa.GaussianBlur(sigma=(1, 2.5))
])

# zoom et dezoom l'image
zoom_picture_aug = iaa.Sequential([
    iaa.Affine(scale= (0.8,1.2))
])

# modifie le contrastre de l'image en pos
contrast_neg_picture_aug = iaa.Sequential([
    iaa.GammaContrast((0.3, 1))
])

# modifie le contrastre de l'image en neg
contrast_pos_picture_aug = iaa.Sequential([
    iaa.GammaContrast((1, 2))
])

# modifie la saturation de l'image
Saturation_picture_aug = iaa.Sequential([
    iaa.RemoveSaturation((0.1,0.8))
])

# modifie la temperature de l'image
Temperature_picture_aug = iaa.Sequential([
    iaa.ChangeColorTemperature((3000,40000))
])

# modifie luminosité de l'image en pos et neg
Brightness_picture_aug = iaa.Sequential([
    iaa.BlendAlpha((0.0, 1.0),foreground=iaa.Add(100),background=iaa.Multiply(0.2))
])

# modifie perspective de l'image
Wrappe_picture_aug = iaa.Sequential([
    iaa.PiecewiseAffine(scale=(0.01,0.05))
])

# reotation l'image en pos et neg
Rotate_picture_aug = iaa.Sequential([
    iaa.pillike.Affine(rotate=(-45, 45), fillcolor=(0, 256))
])

# simule soleil
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
# simule pluie
rain_picture_aug = iaa.Sequential([
    iaa.Rain(speed=(0.1, 0.5))
])

#qualité l'image en pos et neg
qualite_picture_aug23 = iaa.Sequential([
    iaa.BlendAlpha([0.25, 0.75], iaa.MedianBlur(13)),
    iaa.JpegCompression(compression=(50, 90))
])

#bruit l'image en pos et neg
Noise_picture_aug = iaa.Sequential([
    iaa.AdditiveGaussianNoise(loc=0, scale=(30,50), per_channel=0.5)

])

# met des carré/rectange de couleur aléatoirement sur l'image
Cutout_colores_picture_aug = iaa.Sequential([
    iaa.Cutout(fill_mode="constant",nb_iterations=(1, 5), cval=(0, 255),squared=False, fill_per_channel=0.5)
])

# met des carré/rectange en bruit aléatoirement sur l'image
Cutout_noise_picture_aug = iaa.Sequential([
    iaa.Cutout(fill_mode="gaussian", nb_iterations=(1, 5),squared=False, fill_per_channel=True)
])

# met des petits carrés de couleur transparent aléatoirement sur l'image
CoarseDropout_picture_aug256 = iaa.Sequential([
    iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5)
])

# seq = iaa.BlendAlpha(
#     factor=(0.2, 0.8),
#     foreground=iaa.Sharpen(1.0, lightness=2),
#     background=iaa.CoarseDropout(p=0.1, size_px=8)
# )


i = 0
for j in range(3):
    blur_picture = Noise_picture_aug(images=images)
    for img in blur_picture:
        url = 'EiffelTower_picture/Pinterest_modify/img' + str(i) + ".jpg"
        i+=1
        # cv2.imshow("Image", img)
        cv2.imwrite(url, img)
        cv2.waitKey(0)


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