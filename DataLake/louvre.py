from dif import dif

from quality_img import *
from scrape_pinterest import *
from data_augmentation import *
from scrape_IG import *
from subfolders import *

# data augmentaion des immage
# data_augmentation("EiffelTower_picture/Pinterest", "EiffelTower_picture/Pinterest_modify")

# scraping insta loc tour effel
# scrap_insta_loc(user_name="tomghii", password="RH28ins!+",tag_loc="pyramide-du-louvre",directorie_outpute="louvre_picture\Insta", file_link="link_louvre.txt")

# scraping pintereste tour effel
# url = "https://www.pinterest.fr/search/pins/?q=pyramide%20du%20louvre&rs=typed&term_meta[]=pyramide%7Ctyped&term_meta[]=du%7Ctyped&term_meta[]=louvre%7Ctyped"
# web_scraping_Pinterest(url,"louvre_picture\Pinterest","link_louvre.txt")
# url = "https://www.pinterest.fr/search/pins/?q=pyramide%20du%20louvre%20photo&rs=typed&term_meta[]=pyramide%7Ctyped&term_meta[]=du%7Ctyped&term_meta[]=louvre%7Ctyped&term_meta[]=photo%7Ctyped"
# web_scraping_Pinterest(url,"louvre_picture\Pinterest","link_louvre.txt")
# url = "https://www.pinterest.fr/search/pins/?q=pyramide%20du%20louvre%20photo%20touriste&rs=typed&term_meta[]=pyramide%7Ctyped&term_meta[]=du%7Ctyped&term_meta[]=louvre%7Ctyped&term_meta[]=photo%7Ctyped&term_meta[]=touriste%7Ctyped"
# web_scraping_Pinterest(url,"louvre_picture\Pinterest","link_louvre.txt")
# url = "https://www.pinterest.fr/search/pins/?q=pyramide%20du%20louvre%20photo%20nuit&rs=typed&term_meta[]=pyramide%7Ctyped&term_meta[]=du%7Ctyped&term_meta[]=louvre%7Ctyped&term_meta[]=photo%7Ctyped&term_meta[]=nuit%7Ctyped"
# web_scraping_Pinterest(url,"louvre_picture\Pinterest","link_louvre.txt")
# url = "https://www.pinterest.fr/search/pins/?q=pyramide%20du%20louvre%20photography&rs=typed&term_meta[]=pyramide%7Ctyped&term_meta[]=du%7Ctyped&term_meta[]=louvre%7Ctyped&term_meta[]=photography%7Ctyped"
# web_scraping_Pinterest(url,"louvre_picture\Pinterest","link_louvre.txt")
# url = "https://www.pinterest.fr/search/pins/?q=pyramide%20du%20louvre%20photography%20night&rs=typed&term_meta[]=pyramide%7Ctyped&term_meta[]=du%7Ctyped&term_meta[]=louvre%7Ctyped&term_meta[]=photography%7Ctyped&term_meta[]=night%7Ctyped"
# web_scraping_Pinterest(url,"louvre_picture\Pinterest","link_louvre.txt")


# divise en trois sous dossier les img
# divided_into_subfolders("louvre_picture/all_clean")
# divided_into_subfolders("louvre_picture/all_clean/sub1")
# divided_into_subfolders("louvre_picture/all_clean/sub2")
# divided_into_subfolders("louvre_picture/all_clean/sub3")

# modification de la taille et qualité de l'image
modifications_size("louvre_picture/all_clean/sub1/sub1", "louvre_picture/modif/sub1/sub1", 32)
modifications_size("louvre_picture/all_clean/sub1/sub2", "louvre_picture/modif/sub1/sub2", 32)
modifications_size("louvre_picture/all_clean/sub1/sub3", "louvre_picture/modif/sub1/sub3", 32)

modifications_size("louvre_picture/all_clean/sub2/sub1", "louvre_picture/modif/sub2/sub1", 32)
modifications_size("louvre_picture/all_clean/sub2/sub2", "louvre_picture/modif/sub2/sub2", 32)
modifications_size("louvre_picture/all_clean/sub2/sub3", "louvre_picture/modif/sub2/sub3", 32)

modifications_size("louvre_picture/all_clean/sub3/sub1", "louvre_picture/modif/sub3/sub1", 32)
modifications_size("louvre_picture/all_clean/sub3/sub2", "louvre_picture/modif/sub3/sub2", 32)
modifications_size("louvre_picture/all_clean/sub3/sub3", "louvre_picture/modif/sub3/sub3", 32)
#
#
# # recherche de doublon (j'ai trouvé mieux : dupeguru)
# # search = dif("C:/Users/tomca/Documents/ESGI/3IABD2/S2/Projet_Annuel/Projet-Annuel-3IABD/Data Lake/EiffelTower_picture/all/",show_output=True,similarity="normal")
# # print(search.result)

