from recuperation_good_img import *
from quality_img import *
from divise_test_entrainement import *
def run():
    # import eiffel_tower
    # import arc_de_triomphe
    # recup_img("EiffelTower_picture/all_img_triee/", "EiffelTower_picture/32_img_triee/", "EiffelTower_picture/origine_img_triee/")
    # recup_img("arc_de_triomphe_picture/all_img_triee/", "arc_de_triomphe_picture/32_img_triee/", "arc_de_triomphe_picture/origine_img_triee/")
    # modifications_color_BW("arc_de_triomphe_picture/32_img_triee/", "arc_de_triomphe_picture/modif_colors/")
    # modifications_color_BW("EiffelTower_picture/32_img_triee/", "EiffelTower_picture/modif_colors/")
    test_train("EiffelTower_picture/modif_colors/", "EiffelTower_picture/img_test/", "EiffelTower_picture/img_train/")
    test_train("arc_de_triomphe_picture/modif_colors/", "arc_de_triomphe_picture/img_test/", "arc_de_triomphe_picture/img_train/")



if __name__ == "__main__":
    run()