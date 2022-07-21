from recuperation_good_img import *
from quality_img import *
from divise_test_entrainement import *
from download_pictures_link import download
def run():
# importaion des images
    # import eiffel_tower
    # import arc_de_triomphe
    # import louvre
    # import pantheon

# pour 32/DS/max
    # recup_img("EiffelTower_picture/all_img_triee/", "EiffelTower_picture/32_img_triee/", "EiffelTower_picture/origine_img_triee/")
    # modifications_color_BW("EiffelTower_picture/32_img_triee/", "EiffelTower_picture/modif_colors/")
    # test_train("EiffelTower_picture/modif_colors/", "EiffelTower_picture/img_test/", "EiffelTower_picture/img_train/")

    # recup_img("arc_de_triomphe_picture/all_img_triee/", "arc_de_triomphe_picture/32_img_triee/", "arc_de_triomphe_picture/origine_img_triee/")
    # modifications_color_BW("arc_de_triomphe_picture/32_img_triee/", "arc_de_triomphe_picture/modif_colors/")
    # test_train("arc_de_triomphe_picture/modif_colors/", "arc_de_triomphe_picture/img_test/", "arc_de_triomphe_picture/img_train/")

    # recup_img("D:/PA/louvre_picture/all_img_triee/", "D:/PA/louvre_picture/32_img_triee/", "D:/PA/louvre_picture/origine_img_triee/")
    # modifications_color_BW("D:/PA/louvre_picture/32_img_triee/", "D:/PA/louvre_picture/modif_colors/")
    # test_train("D:/PA/louvre_picture/modif_colors/", "D:/PA/louvre_picture/img_test/", "D:/PA/louvre_picture/img_train/")

    # recup_img("D:/PA/pantheon_picture/all_img_triee/", "D:/PA/pantheon_picture/32_img_triee/", "D:/PA/pantheon_picture/origine_img_triee/")
    # modifications_color_BW("D:/PA/pantheon_picture/32_img_triee/", "D:/PA/pantheon_picture/modif_colors/")
    # test_train("D:/PA/pantheon_picture/modif_colors/", "D:/PA/pantheon_picture/img_test/", "D:/PA/pantheon_picture/img_train/")


# /32/64/DS+aug/DS++aug/max
#     for i in range(2):
#         aug("D:/PA/EiffelTower_picture/origine_img_triee", "D:/PA/EiffelTower_picture/aug+/origine/", "_TR" + str(i))
#         aug("D:/PA/EiffelTower_picture/origine_img_triee2", "D:/PA/EiffelTower_picture/aug+/origine/", "_TR" + str(i))
#         aug("D:/PA/EiffelTower_picture/origine_img_triee3", "D:/PA/EiffelTower_picture/aug+/origine/", "_TR" + str(i))
#         aug("D:/PA/EiffelTower_picture/origine_img_triee4", "D:/PA/EiffelTower_picture/aug+/origine/", "_TR" + str(i))
#     for i in range(4):
#         aug("D:/PA/EiffelTower_picture/origine_img_triee", "D:/PA/EiffelTower_picture/aug++/origine/", "_TR" + str(i))
#         aug("D:/PA/EiffelTower_picture/origine_img_triee2", "D:/PA/EiffelTower_picture/aug++/origine/", "_TR" + str(i))
#         aug("D:/PA/EiffelTower_picture/origine_img_triee3", "D:/PA/EiffelTower_picture/aug++/origine/", "_TR" + str(i))
#         aug("D:/PA/EiffelTower_picture/origine_img_triee4", "D:/PA/EiffelTower_picture/aug++/origine/", "_TR" + str(i))
#     modifications_size("D:/PA/EiffelTower_picture/aug+/origine", "D:/PA/EiffelTower_picture/aug+/32/RGB", 32)
#     modifications_size("D:/PA/EiffelTower_picture/aug+/origine", "D:/PA/EiffelTower_picture/aug+/64/RGB", 64)
#     modifications_size("D:/PA/EiffelTower_picture/aug++/origine", "D:/PA/EiffelTower_picture/aug++/32/RGB", 32)
#     modifications_size("D:/PA/EiffelTower_picture/aug++/origine", "D:/PA/EiffelTower_picture/aug++/64/RGB", 64)
#
#     modifications_color_BW("D:/PA/EiffelTower_picture/aug+/32/RGB/", "D:/PA/EiffelTower_picture/aug+/32/NB/")
#     modifications_color_BW("D:/PA/EiffelTower_picture/aug+/64/RGB/", "D:/PA/EiffelTower_picture/aug+/64/NB/")
#     modifications_color_BW("D:/PA/EiffelTower_picture/aug++/32/RGB/", "D:/PA/EiffelTower_picture/aug++/32/NB/")
#     modifications_color_BW("D:/PA/EiffelTower_picture/aug++/64/RGB/", "D:/PA/EiffelTower_picture/aug++/64/NB/")

    # test_train("D:/PA/EiffelTower_picture/aug+/32/NB/", "D:/PA/EiffelTower_picture/img_test_arg+_32", "D:/PA/EiffelTower_picture/img_train_arg+_32")
    # test_train("D:/PA/EiffelTower_picture/aug+/64/NB/", "D:/PA/EiffelTower_picture/img_test_arg+_64","D:/PA/EiffelTower_picture/img_train_arg+_64")
    # test_train("D:/PA/EiffelTower_picture/aug++/32/NB/", "D:/PA/EiffelTower_picture/img_test_arg++_32","D:/PA/EiffelTower_picture/img_train_arg++_32")
    # test_train("D:/PA/EiffelTower_picture/aug++/64/NB/", "D:/PA/EiffelTower_picture/img_test_arg++_64","D:/PA/EiffelTower_picture/img_train_arg++_64")

    # for i in range(2):
    #     aug("D:/PA/arc_de_triomphe_picture/origine_img_triee", "D:/PA/arc_de_triomphe_picture/aug+/origine/", "_TR" + str(i))
    # for i in range(2):
    #     aug("D:/PA/arc_de_triomphe_picture/origine_img_triee", "D:/PA/arc_de_triomphe_picture/aug++/origine/", "_TR" + str(i))
    modifications_size("D:/PA/arc_de_triomphe_picture/aug+/origine", "D:/PA/arc_de_triomphe_picture/aug+/32/RGB", 32)
    modifications_size("D:/PA/arc_de_triomphe_picture/aug+/origine", "D:/PA/arc_de_triomphe_picture/aug+/64/RGB", 64)
    modifications_size("D:/PA/arc_de_triomphe_picture/aug++/origine", "D:/PA/arc_de_triomphe_picture/aug++/32/RGB", 32)
    modifications_size("D:/PA/arc_de_triomphe_picture/aug++/origine", "D:/PA/arc_de_triomphe_picture/aug++/64/RGB", 64)

    modifications_color_BW("D:/PA/arc_de_triomphe_picture/aug+/32/RGB/", "D:/PA/arc_de_triomphe_picture/aug+/32/NB/")
    modifications_color_BW("D:/PA/arc_de_triomphe_picture/aug+/64/RGB/", "D:/PA/arc_de_triomphe_picture/aug+/64/NB/")
    modifications_color_BW("D:/PA/arc_de_triomphe_picture/aug++/32/RGB/", "D:/PA/arc_de_triomphe_picture/aug++/32/NB/")
    modifications_color_BW("D:/PA/arc_de_triomphe_picture/aug++/64/RGB/", "D:/PA/arc_de_triomphe_picture/aug++/64/NB/")

    test_train("D:/PA/arc_de_triomphe_picture/aug+/32/NB/", "D:/PA/arc_de_triomphe_picture/img_test_arg+_32", "D:/PA/arc_de_triomphe_picture/img_train_arg+_32")
    test_train("D:/PA/arc_de_triomphe_picture/aug+/64/NB/", "D:/PA/arc_de_triomphe_picture/img_test_arg+_64", "D:/PA/arc_de_triomphe_picture/img_train_arg+_64")
    test_train("D:/PA/arc_de_triomphe_picture/aug++/32/NB/", "D:/PA/arc_de_triomphe_picture/img_test_arg++_32", "D:/PA/arc_de_triomphe_picture/img_train_arg++_32")
    test_train("D:/PA/arc_de_triomphe_picture/aug++/64/NB/", "D:/PA/arc_de_triomphe_picture/img_test_arg++_64", "D:/PA/arc_de_triomphe_picture/img_train_arg++_64")

    # for i in range(2):
    #     aug("D:/PA/louvre_picture/origine_img_triee", "D:/PA/louvre_picture/aug+/origine/", "_TR" + str(i))
    # for i in range(4):
    #     aug("D:/PA/louvre_picture/origine_img_triee", "D:/PA/louvre_picture/aug++/origine/", "_TR" + str(i))
    # modifications_size("D:/PA/louvre_picture/aug+/origine", "D:/PA/louvre_picture/aug+/32/RGB", 32)
    # modifications_size("D:/PA/louvre_picture/aug+/origine", "D:/PA/louvre_picture/aug+/64/RGB", 64)
    # modifications_size("D:/PA/louvre_picture/aug++/origine", "D:/PA/louvre_picture/aug++/32/RGB", 32)
    # modifications_size("D:/PA/louvre_picture/aug++/origine", "D:/PA/louvre_picture/aug++/64/RGB", 64)
    #
    # modifications_color_BW("D:/PA/louvre_picture/aug+/32/RGB/", "D:/PA/louvre_picture/aug+/32/NB/")
    # modifications_color_BW("D:/PA/louvre_picture/aug+/64/RGB/", "D:/PA/louvre_picture/aug+/64/NB/")
    # modifications_color_BW("D:/PA/louvre_picture/aug++/32/RGB/", "D:/PA/louvre_picture/aug++/32/NB/")
    # modifications_color_BW("D:/PA/louvre_picture/aug++/64/RGB/", "D:/PA/louvre_picture/aug++/64/NB/")
    #
    # test_train("D:/PA/louvre_picture/aug+/32/NB/", "D:/PA/louvre_picture/img_test_arg+_32", "D:/PA/louvre_picture/img_train_arg+_32")
    # test_train("D:/PA/louvre_picture/aug+/32/NB/", "D:/PA/louvre_picture/img_test_arg+_64", "D:/PA/louvre_picture/img_train_arg+_64")
    # test_train("D:/PA/louvre_picture/aug+/32/NB/", "D:/PA/louvre_picture/img_test_arg++_32", "D:/PA/louvre_picture/img_train_arg++_32")
    # test_train("D:/PA/louvre_picture/aug+/32/NB/", "D:/PA/louvre_picture/img_test_arg++_64", "D:/PA/louvre_picture/img_train_arg++_64")
    #
    # for i in range(2):
    #     aug("D:/PA/pantheon_picture/origine_img_triee", "D:/PA/pantheon_picture/aug+/origine/", "_TR" + str(i))
    # for i in range(4):
    #     aug("D:/PA/pantheon_picture/origine_img_triee", "D:/PA/pantheon_picture/aug++/origine/", "_TR" + str(i))
    # modifications_size("D:/PA/pantheon_picture/aug+/origine", "D:/PA/pantheon_picture/aug+/32/RGB", 32)
    # modifications_size("D:/PA/pantheon_picture/aug+/origine", "D:/PA/pantheon_picture/aug+/64/RGB", 64)
    # modifications_size("D:/PA/pantheon_picture/aug++/origine", "D:/PA/pantheon_picture/aug++/32/RGB", 32)
    # modifications_size("D:/PA/pantheon_picture/aug++/origine", "D:/PA/pantheon_picture/aug++/64/RGB", 64)
    #
    # modifications_color_BW("D:/PA/pantheon_picture/aug+/32/RGB/", "D:/PA/pantheon_picture/aug+/32/NB/")
    # modifications_color_BW("D:/PA/pantheon_picture/aug+/64/RGB/", "D:/PA/pantheon_picture/aug+/64/NB/")
    # modifications_color_BW("D:/PA/pantheon_picture/aug++/32/RGB/", "D:/PA/pantheon_picture/aug++/32/NB/")
    # modifications_color_BW("D:/PA/pantheon_picture/aug++/64/RGB/", "D:/PA/pantheon_picture/aug++/64/NB/")
    #
    # test_train("D:/PA/pantheon_picture/aug+/32/NB/", "D:/PA/pantheon_picture/img_test_arg+_32", "D:/PA/pantheon_picture/img_train_arg+_32")
    # test_train("D:/PA/pantheon_picture/aug+/32/NB/", "D:/PA/pantheon_picture/img_test_arg+_64", "D:/PA/pantheon_picture/img_train_arg+_64")
    # test_train("D:/PA/pantheon_picture/aug+/32/NB/", "D:/PA/pantheon_picture/img_test_arg++_32", "D:/PA/pantheon_picture/img_train_arg++_32")
    # test_train("D:/PA/pantheon_picture/aug+/32/NB/", "D:/PA/pantheon_picture/img_test_arg++_64", "D:/PA/pantheon_picture/img_train_arg++_64")

# img aurigine 64
#     modifications_size("D:/PA/EiffelTower_picture/origine_img_triee", "D:/PA/EiffelTower_picture/64_img_triee", 64)
#     modifications_size("D:/PA/arc_de_triomphe_picture/origine_img_triee", "D:/PA/arc_de_triomphe_picture/64_img_triee", 64)
#     modifications_size("D:/PA/louvre_picture/origine_img_triee", "D:/PA/louvre_picture/64_img_triee", 64)
#     modifications_size("D:/PA/pantheon_picture/origine_img_triee", "D:/PA/pantheon_picture/64_img_triee", 64)
#
#     modifications_color_BW("D:/PA/EiffelTower_picture/64_img_triee/", "D:/PA/EiffelTower_picture/64_modif_colors/")
#     modifications_color_BW("D:/PA/arc_de_triomphe_picture/64_img_triee/", "D:/PA/arc_de_triomphe_picture/64_modif_colors/")
#     modifications_color_BW("D:/PA/louvre_picture/64_img_triee/", "D:/PA/louvre_picture/64_modif_colors/")
#     modifications_color_BW("D:/PA/pantheon_picture/64_img_triee/", "D:/PA/pantheon_picture/64_modif_colors/")
#
#     test_train("D:/PA/EiffelTower_picture/64_modif_colors/", "D:/PA/pantheon_picture/64_img_test/", "D:/PA/pantheon_picture/64_img_test/")
#     test_train("D:/PA/pantheon_picture/64_modif_colors/", "D:/PA/pantheon_picture/64_img_test/", "D:/PA/pantheon_picture/64_img_test/")
#     test_train("D:/PA/pantheon_picture/64_modif_colors/", "D:/PA/pantheon_picture/64_img_test/", "D:/PA/pantheon_picture/64_img_test/")
#     test_train("D:/PA/pantheon_picture/64_modif_colors/", "D:/PA/pantheon_picture/64_img_test/", "D:/PA/pantheon_picture/64_img_test/")

if __name__ == "__main__":
    run()