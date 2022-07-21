from dif import dif

from quality_img import *
from scrape_pinterest import *
from data_augmentation import *
from scrape_IG import *
from subfolders import *

# data augmentaion des immage
# data_augmentation("EiffelTower_picture/Pinterest", "EiffelTower_picture/Pinterest_modify")

# scraping insta loc tour effel
# scrap_insta_loc(user_name="tomghii", password="RH28ins!+",tag_loc="pantheonparis",directorie_outpute="pantheon_picture\Insta", file_link="link_pantheon.txt")
# scrap_insta_loc(user_name="tomghii", password="RH28ins!+",tag_loc="paris-pantheon-parade",directorie_outpute="pantheon_picture\Insta", file_link="link_pantheon.txt")

# scraping pintereste tour effel
# url = "https://www.pinterest.fr/search/pins/?q=paris%20pantheon&rs=typed&term_meta[]=paris%7Ctyped&term_meta[]=pantheon%7Ctyped"
# web_scraping_Pinterest(url,"pantheon_picture\Pinterest","link_pantheon.txt")


# divise en trois sous dossier les img
# divided_into_subfolders("pantheon_picture/all_clean")
# divided_into_subfolders("pantheon_picture/all_clean/sub1")
# divided_into_subfolders("pantheon_picture/all_clean/sub2")
# divided_into_subfolders("pantheon_picture/all_clean/sub3")
# #
# modification de la taille et qualité de l'image
modifications_size("pantheon_picture/all_clean/sub1/sub1", "pantheon_picture/modif/sub1/sub1/", 32)
modifications_size("pantheon_picture/all_clean/sub1/sub2", "pantheon_picture/modif/sub1/sub2", 32)
modifications_size("pantheon_picture/all_clean/sub1/sub3", "pantheon_picture/modif/sub1/sub3", 32)


modifications_size("pantheon_picture/all_clean/sub2/sub1", "pantheon_picture/modif/sub2/sub1", 32)
modifications_size("pantheon_picture/all_clean/sub2/sub2", "pantheon_picture/modif/sub2/sub2", 32)
modifications_size("pantheon_picture/all_clean/sub2/sub3", "pantheon_picture/modif/sub2/sub3", 32)

modifications_size("pantheon_picture/all_clean/sub3/sub1", "pantheon_picture/modif/sub3/sub1", 32)
modifications_size("pantheon_picture/all_clean/sub3/sub2", "pantheon_picture/modif/sub3/sub2", 32)
modifications_size("pantheon_picture/all_clean/sub3/sub3", "pantheon_picture/modif/sub3/sub3", 32)


