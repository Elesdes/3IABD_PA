# img aurigine 64
from quality_img import *
from divise_test_entrainement import *
def run():
    # for i in range(2):
    #     aug("D:/PA/arc_de_triomphe_picture/origine_img_triee", "D:/PA/arc_de_triomphe_picture/aug+/origine/", "_TR" + str(i))
    for i in range(2):
        aug("D:/PA/arc_de_triomphe_picture/origine_img_triee", "D:/PA/arc_de_triomphe_picture/aug++/origine/", "_TR" + str(i+2))
    # modifications_size("D:/PA/arc_de_triomphe_picture/aug+/origine", "D:/PA/arc_de_triomphe_picture/aug+/32/RGB", 32)
    # modifications_size("D:/PA/arc_de_triomphe_picture/aug+/origine", "D:/PA/arc_de_triomphe_picture/aug+/64/RGB", 64)
    # modifications_size("D:/PA/arc_de_triomphe_picture/aug++/origine", "D:/PA/arc_de_triomphe_picture/aug++/32/RGB", 32)
    # modifications_size("D:/PA/arc_de_triomphe_picture/aug++/origine", "D:/PA/arc_de_triomphe_picture/aug++/64/RGB", 64)
    #
    # modifications_color_BW("D:/PA/arc_de_triomphe_picture/aug+/32/RGB/", "D:/PA/arc_de_triomphe_picture/aug+/32/NB/")
    # modifications_color_BW("D:/PA/arc_de_triomphe_picture/aug+/64/RGB/", "D:/PA/arc_de_triomphe_picture/aug+/64/NB/")
    # modifications_color_BW("D:/PA/arc_de_triomphe_picture/aug++/32/RGB/", "D:/PA/arc_de_triomphe_picture/aug++/32/NB/")
    # modifications_color_BW("D:/PA/arc_de_triomphe_picture/aug++/64/RGB/", "D:/PA/arc_de_triomphe_picture/aug++/64/NB/")
    #
    # test_train("D:/PA/arc_de_triomphe_picture/aug+/32/NB/", "D:/PA/arc_de_triomphe_picture/img_test_arg+_32", "D:/PA/arc_de_triomphe_picture/img_train_arg+_32")
    # test_train("D:/PA/arc_de_triomphe_picture/aug+/64/NB/", "D:/PA/arc_de_triomphe_picture/img_test_arg+_64", "D:/PA/arc_de_triomphe_picture/img_train_arg+_64")
    # test_train("D:/PA/arc_de_triomphe_picture/aug++/32/NB/", "D:/PA/arc_de_triomphe_picture/img_test_arg++_32", "D:/PA/arc_de_triomphe_picture/img_train_arg++_32")
    # test_train("D:/PA/arc_de_triomphe_picture/aug++/64/NB/", "D:/PA/arc_de_triomphe_picture/img_test_arg++_64", "D:/PA/arc_de_triomphe_picture/img_train_arg++_64")

if __name__ == "__main__":
    run()