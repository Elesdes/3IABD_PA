from dif import dif

from quality_img import *
from scrape_pinterest import *
from data_augmentation import *
from scrape_IG import *
from subfolders import *

# data augmentaion des immage
data_augmentation("EiffelTower_picture/Pinterest", "EiffelTower_picture/Pinterest_modify")

# scraping insta loc tour effel
scrap_insta_loc(user_name="tomghii", password="",tag_loc="tour-eiffel",directorie_outpute="EiffelTower_picture\Insta")

# scraping pintereste tour effel
url = "https://www.pinterest.fr/search/pins/?q=Tour%20Eiffel&rs=typed&term_meta[]=Tour%7Ctyped&term_meta[]=Eiffel%7Ctyped"
# web_scraping_Pinterest(url)
url = "https://www.pinterest.fr/search/pins/?q=Tour%20Eiffel%20photo&rs=typed&term_meta[]=Tour%7Ctyped&term_meta[]=Eiffel%7Ctyped&term_meta[]=photo%7Ctyped"
# web_scraping_Pinterest(url)
url = "https://www.pinterest.fr/search/pins/?q=Tour%20Eiffel%20photo%20touriste&rs=typed&term_meta[]=Tour%7Ctyped&term_meta[]=Eiffel%7Ctyped&term_meta[]=photo%7Ctyped&term_meta[]=touriste%7Ctyped"
# web_scraping_Pinterest(url)
url = "https://www.pinterest.fr/search/pins/?rs=ac&len=2&q=tour%20eiffel%20photo%20nuit&eq=Tour%20Eiffel%20photo%20n&etslf=6150&term_meta[]=tour%7Cautocomplete%7C1&term_meta[]=eiffel%7Cautocomplete%7C1&term_meta[]=photo%7Cautocomplete%7C1&term_meta[]=nuit%7Cautocomplete%7C1"
# web_scraping_Pinterest(url)
url = "https://www.pinterest.fr/search/pins/?rs=ac&len=2&q=tour%20eiffel%20photography&eq=tour%20eiffel%20photo&etslf=12887&term_meta[]=tour%7Cautocomplete%7C1&term_meta[]=eiffel%7Cautocomplete%7C1&term_meta[]=photography%7Cautocomplete%7C1"
# web_scraping_Pinterest(url)
url = "https://www.pinterest.fr/search/pins/?rs=ac&len=2&q=tour%20eiffel%20photography%20night&eq=tour%20eiffel%20photography%20n&etslf=5541&term_meta[]=tour%7Cautocomplete%7C0&term_meta[]=eiffel%7Cautocomplete%7C0&term_meta[]=photography%7Cautocomplete%7C0&term_meta[]=night%7Cautocomplete%7C0"
# web_scraping_Pinterest(url)

# modification de la taille et qualité de l'image
modifications_size("EiffelTower_picture/all_clean/sub1", "EiffelTower_picture/modif/sub1/", 32)
modifications_size("EiffelTower_picture/all_clean/sub2", "EiffelTower_picture/modif/sub2/", 32)
modifications_size("EiffelTower_picture/all_clean/sub3", "EiffelTower_picture/modif/sub3/", 32)

# divise en trois sous dossier les img
divided_into_subfolders("EiffelTower_picture/all_clean")


# recherche de doublon (j'ai trouvé mieux : dupeguru)
# search = dif("C:/Users/tomca/Documents/ESGI/3IABD2/S2/Projet_Annuel/Projet-Annuel-3IABD/Data Lake/EiffelTower_picture/all/",show_output=True,similarity="normal")
# print(search.result)

