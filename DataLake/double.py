# img aurigine 64
from quality_img import *
from divise_test_entrainement import *
def run():
    # modifications_size("D:/PA/EiffelTower_picture/origine_img_triee", "D:/PA/EiffelTower_picture/64_img_triee", 64)
    # modifications_size("D:/PA/EiffelTower_picture/origine_img_triee2", "D:/PA/EiffelTower_picture/64_img_triee", 64)
    # modifications_size("D:/PA/EiffelTower_picture/origine_img_triee3", "D:/PA/EiffelTower_picture/64_img_triee", 64)
    # modifications_size("D:/PA/EiffelTower_picture/origine_img_triee4", "D:/PA/EiffelTower_picture/64_img_triee", 64)
    # modifications_size("D:/PA/arc_de_triomphe_picture/origine_img_triee", "D:/PA/arc_de_triomphe_picture/64_img_triee", 64)
    # modifications_size("D:/PA/louvre_picture/origine_img_triee", "D:/PA/louvre_picture/64_img_triee", 64)
    # modifications_size("D:/PA/pantheon_picture/origine_img_triee", "D:/PA/pantheon_picture/64_img_triee", 64)

    # modifications_color_BW("D:/PA/EiffelTower_picture/64_img_triee/", "D:/PA/EiffelTower_picture/64_modif_colors/")
    # modifications_color_BW("D:/PA/arc_de_triomphe_picture/64_img_triee/", "D:/PA/arc_de_triomphe_picture/64_modif_colors/")
    # modifications_color_BW("D:/PA/louvre_picture/64_img_triee/", "D:/PA/louvre_picture/64_modif_colors/")
    # modifications_color_BW("D:/PA/pantheon_picture/64_img_triee/", "D:/PA/pantheon_picture/64_modif_colors/")

    test_train("D:/PA/EiffelTower_picture/64_modif_colors/", "D:/PA/EiffelTower_picture/64_img_test/", "D:/PA/EiffelTower_picture/64_img_train/")
    test_train("D:/PA/arc_de_triomphe_picture/64_modif_colors/", "D:/PA/arc_de_triomphe_picture/64_img_test/", "D:/PA/arc_de_triomphe_picture/64_img_train/")
    test_train("D:/PA/louvre_picture/64_modif_colors/", "D:/PA/louvre_picture/64_img_test/", "D:/PA/louvre_picture/64_img_train/")
    test_train("D:/PA/pantheon_picture/64_modif_colors/", "D:/PA/pantheon_picture/64_img_test/", "D:/PA/pantheon_picture/64_img_train/")

if __name__ == "__main__":
    run()