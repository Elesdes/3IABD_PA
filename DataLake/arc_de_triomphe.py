from dif import dif

from quality_img import *
from scrape_pinterest import *
from data_augmentation import *
from scrape_IG import *
from subfolders import *

# data augmentaion des immage
# data_augmentation("EiffelTower_picture/Pinterest", "EiffelTower_picture/Pinterest_modify")

# scraping insta loc tour effel
# scrap_insta_loc(user_name="tomghii", password="",tag_loc="arc-de-triomphe",directorie_outpute="arc_de_triomphe_picture\Insta", file_link="link_arc_de_triomphe.txt")

# scraping pintereste tour effel
# url = "https://www.pinterest.fr/search/pins/?q=arc%20de%20triomphe&rs=typed&term_meta[]=arc%7Ctyped&term_meta[]=de%7Ctyped&term_meta[]=triomphe%7Ctyped"
# web_scraping_Pinterest(url,"arc_de_triomphe_picture\Pinterest","link_arc_de_triomphe.txt")
# url = "https://www.pinterest.fr/search/pins/?q=arc%20de%20triomphe%20photo&rs=typed&term_meta[]=arc%7Ctyped&term_meta[]=de%7Ctyped&term_meta[]=triomphe%7Ctyped&term_meta[]=photo%7Ctyped"
# web_scraping_Pinterest(url,"arc_de_triomphe_picture\Pinterest","link_arc_de_triomphe.txt")
# url = "https://www.pinterest.fr/search/pins/?q=arc%20de%20triomphe%20photo%20touriste&rs=typed&term_meta[]=arc%7Ctyped&term_meta[]=de%7Ctyped&term_meta[]=triomphe%7Ctyped&term_meta[]=photo%7Ctyped&term_meta[]=touriste%7Ctyped"
# web_scraping_Pinterest(url,"arc_de_triomphe_picture\Pinterest","link_arc_de_triomphe.txt")
# url = "https://www.pinterest.fr/search/pins/?q=arc%20de%20triomphe%20photo%20nuit&rs=typed&term_meta[]=arc%7Ctyped&term_meta[]=de%7Ctyped&term_meta[]=triomphe%7Ctyped&term_meta[]=photo%7Ctyped&term_meta[]=nuit%7Ctyped"
# web_scraping_Pinterest(url,"arc_de_triomphe_picture\Pinterest","link_arc_de_triomphe.txt")
# url = "https://www.pinterest.fr/search/pins/?rs=ac&len=2&q=arc%20de%20triomphe%20photography&eq=arc%20de%20triomphe%20photogr&etslf=4322&term_meta[]=arc%7Cautocomplete%7C0&term_meta[]=de%7Cautocomplete%7C0&term_meta[]=triomphe%7Cautocomplete%7C0&term_meta[]=photography%7Cautocomplete%7C0"
# web_scraping_Pinterest(url,"arc_de_triomphe_picture\Pinterest","link_arc_de_triomphe.txt")
# url = "https://www.pinterest.fr/search/pins/?rs=ac&len=2&q=arc%20de%20triomphe%20photography%20night&eq=arc%20de%20triomphe%20photography&etslf=2811&term_meta[]=arc%7Cautocomplete%7C3&term_meta[]=de%7Cautocomplete%7C3&term_meta[]=triomphe%7Cautocomplete%7C3&term_meta[]=photography%7Cautocomplete%7C3&term_meta[]=night%7Cautocomplete%7C3"
# web_scraping_Pinterest(url,"arc_de_triomphe_picture\Pinterest","link_arc_de_triomphe.txt")


# divise en trois sous dossier les img
# divided_into_subfolders("arc_de_triomphe_picture/all_clean")
# divided_into_subfolders("arc_de_triomphe_picture/all_clean/sub1")
# divided_into_subfolders("arc_de_triomphe_picture/all_clean/sub2")
# divided_into_subfolders("arc_de_triomphe_picture/all_clean/sub3")
#
# # modification de la taille et qualité de l'image
modifications_size("arc_de_triomphe_picture/all_clean/sub1/sub1", "arc_de_triomphe_picture/modif/sub1/sub1/", 32)
modifications_size("arc_de_triomphe_picture/all_clean/sub1/sub2", "arc_de_triomphe_picture/modif/sub1/sub2", 32)
modifications_size("arc_de_triomphe_picture/all_clean/sub1/sub3", "arc_de_triomphe_picture/modif/sub1/sub3", 32)


modifications_size("arc_de_triomphe_picture/all_clean/sub2/sub1", "arc_de_triomphe_picture/modif/sub2/sub1", 32)
modifications_size("arc_de_triomphe_picture/all_clean/sub2/sub2", "arc_de_triomphe_picture/modif/sub2/sub2", 32)
modifications_size("arc_de_triomphe_picture/all_clean/sub2/sub3", "arc_de_triomphe_picture/modif/sub2/sub3", 32)

modifications_size("arc_de_triomphe_picture/all_clean/sub3/sub1", "arc_de_triomphe_picture/modif/sub3/sub1", 32)
modifications_size("arc_de_triomphe_picture/all_clean/sub3/sub2", "arc_de_triomphe_picture/modif/sub3/sub2", 32)
modifications_size("arc_de_triomphe_picture/all_clean/sub3/sub3", "arc_de_triomphe_picture/modif/sub3/sub3", 32)
#
#
# # recherche de doublon (j'ai trouvé mieux : dupeguru)
# # search = dif("C:/Users/tomca/Documents/ESGI/3IABD2/S2/Projet_Annuel/Projet-Annuel-3IABD/Data Lake/EiffelTower_picture/all/",show_output=True,similarity="normal")
# # print(search.result)

