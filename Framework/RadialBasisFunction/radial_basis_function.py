import ctypes as ct
import os
from typing import Any

import PIL
import numpy as np
from PIL.Image import Image

# _32_classique_max
from tqdm import tqdm

EIFFEL_TOWER_TRAINING_32_classique_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS/max/eiffel_tower/img_train/"
EIFFEL_TOWER_TESTING_32_classique_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS/max/eiffel_tower/img_test/"
TRIUMPHAL_ARC_TRAINING_32_classique_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS/max/arc_de_triomphe/img_train/"
TRIUMPHAL_ARC_TESTING_32_classique_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS/max/arc_de_triomphe/img_test/"
LOUVRE_TRAINING_32_classique_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS/max/louvre/img_train/"
LOUVRE_TESTING_32_classique_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS/max/louvre/img_test/"
PANTHEON_TRAINING_32_classique_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS/max/pantheon/img_train/"
PANTHEON_TESTING_32_classique_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS/max/pantheon/img_test/"
# _32_arg_plus_max
EIFFEL_TOWER_TRAINING_32_arg_plus_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS+aug/max/eiffel_tower/img_train_arg+_32/"
EIFFEL_TOWER_TESTING_32_arg_plus_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS+aug/max/eiffel_tower/img_test_arg+_32/"
TRIUMPHAL_ARC_TRAINING_32_arg_plus_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS+aug/max/arc_de_triomphe/img_train_arg+_32/"
TRIUMPHAL_ARC_TESTING_32_arg_plus_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS+aug/max/arc_de_triomphe/img_test_arg+_32/"
LOUVRE_TRAINING_32_arg_plus_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS+aug/max/louvre/img_train_arg+_32/"
LOUVRE_TESTING_32_arg_plus_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS+aug/max/louvre/img_test_arg+_32/"
PANTHEON_TRAINING_32_arg_plus_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS+aug/max/pantheon/img_train_arg+_32/"
PANTHEON_TESTING_32_arg_plus_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS+aug/max/pantheon/img_test_arg+_32/"
# _32_arg_plus_plus_max
EIFFEL_TOWER_TRAINING_32_arg_plus_plus_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS++aug/max/eiffel_tower/img_train_arg++_32/"
EIFFEL_TOWER_TESTING_32_arg_plus_plus_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS++aug/max/eiffel_tower/img_test_arg++_32/"
TRIUMPHAL_ARC_TRAINING_32_arg_plus_plus_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS++aug/max/arc_de_triomphe/img_train_arg++_32/"
TRIUMPHAL_ARC_TESTING_32_arg_plus_plus_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS++aug/max/arc_de_triomphe/img_test_arg++_32/"
LOUVRE_TRAINING_32_arg_plus_plus_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS++aug/max/louvre/img_train_arg++_32/"
LOUVRE_TESTING_32_arg_plus_plus_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS++aug/max/louvre/img_test_arg++_32/"
PANTHEON_TRAINING_32_arg_plus_plus_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS++aug/max/pantheon/img_train_arg++_32/"
PANTHEON_TESTING_32_arg_plus_plus_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS++aug/max/pantheon/img_test_arg++_32/"

# _64_classique_max
EIFFEL_TOWER_TRAINING_64_classique_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS/max/eiffel_tower/64_img_train/"
EIFFEL_TOWER_TESTING_64_classique_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS/max/eiffel_tower/64_img_train/"
TRIUMPHAL_ARC_TRAINING_64_classique_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS/max/arc_de_triomphe/64_img_train/"
TRIUMPHAL_ARC_TESTING_64_classique_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS/max/arc_de_triomphe/64_img_train/"
LOUVRE_TRAINING_64_classique_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS/max/louvre/64_img_train/"
LOUVRE_TESTING_64_classique_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS/max/louvre/64_img_train/"
PANTHEON_TRAINING_64_classique_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS/max/pantheon/64_img_train/"
PANTHEON_TESTING_64_classique_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS/max/pantheon/64_img_train/"
# _64_arg_plus_max
EIFFEL_TOWER_TRAINING_64_arg_plus_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS+aug/max/eiffel_tower/img_train_arg+_64/"
EIFFEL_TOWER_TESTING_64_arg_plus_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS+aug/max/eiffel_tower/img_test_arg+_64/"
TRIUMPHAL_ARC_TRAINING_64_arg_plus_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS+aug/max/arc_de_triomphe/img_train_arg+_64/"
TRIUMPHAL_ARC_TESTING_64_arg_plus_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS+aug/max/arc_de_triomphe/img_test_arg+_64/"
LOUVRE_TRAINING_64_arg_plus_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS+aug/max/louvre/img_train_arg+_64/"
LOUVRE_TESTING_64_arg_plus_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS+aug/max/louvre/img_test_arg+_64/"
PANTHEON_TRAINING_64_arg_plus_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS+aug/max/pantheon/img_train_arg+_64/"
PANTHEON_TESTING_64_arg_plus_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS+aug/max/pantheon/img_test_arg+_64/"
# _64_arg_plus_plus_max
EIFFEL_TOWER_TRAINING_64_arg_plus_plus_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS++aug/max/eiffel_tower/img_train_arg++_64/"
EIFFEL_TOWER_TESTING_64_arg_plus_plus_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS++aug/max/eiffel_tower/img_test_arg++_64/"
TRIUMPHAL_ARC_TRAINING_64_arg_plus_plus_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS++aug/max/arc_de_triomphe/img_train_arg++_64/"
TRIUMPHAL_ARC_TESTING_64_arg_plus_plus_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS++aug/max/arc_de_triomphe/img_test_arg++_64/"
LOUVRE_TRAINING_64_arg_plus_plus_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS++aug/max/louvre/img_train_arg++_64/"
LOUVRE_TESTING_64_arg_plus_plus_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS++aug/max/louvre/img_test_arg++_64/"
PANTHEON_TRAINING_64_arg_plus_plus_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS++aug/max/pantheon/img_train_arg++_64/"
PANTHEON_TESTING_64_arg_plus_plus_max = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS++aug/max/pantheon/img_test_arg++_64/"

# 32_classique_min
EIFFEL_TOWER_TRAINING_32_classique_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS/min/eiffel_tower/img_train/"
EIFFEL_TOWER_TESTING_32_classique_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS/min/eiffel_tower/img_test/"
TRIUMPHAL_ARC_TRAINING_32_classique_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS/min/arc_de_triomphe/img_train/"
TRIUMPHAL_ARC_TESTING_32_classique_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS/min/arc_de_triomphe/img_test/"
LOUVRE_TRAINING_32_classique_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS/min/louvre/img_train/"
LOUVRE_TESTING_32_classique_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS/min/louvre/img_test/"
PANTHEON_TRAINING_32_classique_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS/min/pantheon/img_train/"
PANTHEON_TESTING_32_classique_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS/min/pantheon/img_test/"
# _32_arg_plus_min
EIFFEL_TOWER_TRAINING_32_arg_plus_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS+aug/min/eiffel_tower/img_train_arg+_32/"
EIFFEL_TOWER_TESTING_32_arg_plus_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS+aug/min/eiffel_tower/img_test_arg+_32/"
TRIUMPHAL_ARC_TRAINING_32_arg_plus_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS+aug/min/arc_de_triomphe/img_train_arg+_32/"
TRIUMPHAL_ARC_TESTING_32_arg_plus_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS+aug/min/arc_de_triomphe/img_test_arg+_32/"
LOUVRE_TRAINING_32_arg_plus_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS+aug/min/louvre/img_train_arg+_32/"
LOUVRE_TESTING_32_arg_plus_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS+aug/min/louvre/img_test_arg+_32/"
PANTHEON_TRAINING_32_arg_plus_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS+aug/min/pantheon/img_train_arg+_32/"
PANTHEON_TESTING_32_arg_plus_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS+aug/min/pantheon/img_test_arg+_32/"
# _32_arg_plus_plus_min
EIFFEL_TOWER_TRAINING_32_arg_plus_plus_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS++aug/min/eiffel_tower/img_train_arg++_32/"
EIFFEL_TOWER_TESTING_32_arg_plus_plus_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS++aug/min/eiffel_tower/img_test_arg++_32/"
TRIUMPHAL_ARC_TRAINING_32_arg_plus_plus_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS++aug/min/arc_de_triomphe/img_train_arg++_32/"
TRIUMPHAL_ARC_TESTING_32_arg_plus_plus_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS++aug/min/arc_de_triomphe/img_test_arg++_32/"
LOUVRE_TRAINING_32_arg_plus_plus_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS++aug/min/louvre/img_train_arg++_32/"
LOUVRE_TESTING_32_arg_plus_plus_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS++aug/min/louvre/img_test_arg++_32/"
PANTHEON_TRAINING_32_arg_plus_plus_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS++aug/min/pantheon/img_train_arg++_32/"
PANTHEON_TESTING_32_arg_plus_plus_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/32/DS++aug/min/pantheon/img_test_arg++_32/"

# 64_classique_min
EIFFEL_TOWER_TRAINING_64_classique_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS/min/eiffel_tower/64_img_train/"
EIFFEL_TOWER_TESTING_64_classique_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS/min/eiffel_tower/64_img_train/"
TRIUMPHAL_ARC_TRAINING_64_classique_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS/min/arc_de_triomphe/64_img_train/"
TRIUMPHAL_ARC_TESTING_64_classique_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS/min/arc_de_triomphe/64_img_train/"
LOUVRE_TRAINING_64_classique_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS/min/louvre/64_img_train/"
LOUVRE_TESTING_64_classique_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS/min/louvre/64_img_train/"
PANTHEON_TRAINING_64_classique_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS/min/pantheon/64_img_train/"
PANTHEON_TESTING_64_classique_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS/min/pantheon/64_img_train/"
# _64_arg_plus_min
EIFFEL_TOWER_TRAINING_64_arg_plus_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS+aug/min/eiffel_tower/img_train_arg+_64/"
EIFFEL_TOWER_TESTING_64_arg_plus_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS+aug/min/eiffel_tower/img_test_arg+_64/"
TRIUMPHAL_ARC_TRAINING_64_arg_plus_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS+aug/min/arc_de_triomphe/img_train_arg+_64/"
TRIUMPHAL_ARC_TESTING_64_arg_plus_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS+aug/min/arc_de_triomphe/img_test_arg+_64/"
LOUVRE_TRAINING_64_arg_plus_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS+aug/min/louvre/img_train_arg+_64/"
LOUVRE_TESTING_64_arg_plus_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS+aug/min/louvre/img_test_arg+_64/"
PANTHEON_TRAINING_64_arg_plus_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS+aug/min/pantheon/img_train_arg+_64/"
PANTHEON_TESTING_64_arg_plus_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS+aug/min/pantheon/img_test_arg+_64/"
# _64_arg_plus_plus_min
EIFFEL_TOWER_TRAINING_64_arg_plus_plus_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS++aug/min/eiffel_tower/img_train_arg++_64/"
EIFFEL_TOWER_TESTING_64_arg_plus_plus_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS++aug/min/eiffel_tower/img_test_arg++_64/"
TRIUMPHAL_ARC_TRAINING_64_arg_plus_plus_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS++aug/min/arc_de_triomphe/img_train_arg++_64/"
TRIUMPHAL_ARC_TESTING_64_arg_plus_plus_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS++aug/min/arc_de_triomphe/img_test_arg++_64/"
LOUVRE_TRAINING_64_arg_plus_plus_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS++aug/min/louvre/img_train_arg++_64/"
LOUVRE_TESTING_64_arg_plus_plus_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS++aug/min/louvre/img_test_arg++_64/"
PANTHEON_TRAINING_64_arg_plus_plus_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS++aug/min/pantheon/img_train_arg++_64/"
PANTHEON_TESTING_64_arg_plus_plus_min = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/CompleteDataSet/DS complet_v2/64/DS++aug/min/pantheon/img_test_arg++_64/"

IMAGES_TEST = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/Dataset/DS complet_v2/Tests"

RBF_LIB_PATH = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/Framework/RadialBasisFunction/cmake-build-debug/libRBF.dll"


# Initialize dll for C++ / Python Interop
def init_dll(rbf_lib: ct.CDLL) -> ct.CDLL:
    rbf_lib.test.argtypes = None
    rbf_lib.test.restype = ct.c_int32

    rbf_lib.saveModelRBF.argtypes = [ct.POINTER(ct.POINTER(ct.c_float)), ct.POINTER(ct.c_char), ct.c_int32, ct.c_int32,
                                     ct.c_float]
    rbf_lib.saveModelRBF.restype = None

    rbf_lib.loadModelRBF.argtypes = [ct.POINTER(ct.c_char), ct.c_int32]
    rbf_lib.loadModelRBF.restype = ct.POINTER(ct.POINTER(ct.c_float))

    rbf_lib.destroyFloatArray.argtypes = [ct.POINTER(ct.POINTER(ct.c_float)), ct.c_int32]
    rbf_lib.destroyFloatArray.restype = None

    rbf_lib.flat.argtypes = [ct.POINTER(ct.POINTER(ct.c_float)), ct.c_int32, ct.c_int32]
    rbf_lib.flat.restype = ct.POINTER(ct.POINTER(ct.c_float))

    rbf_lib.initModelWeights.argtypes = [ct.c_int32, ct.c_int32]
    rbf_lib.initModelWeights.restype = ct.POINTER(ct.POINTER(ct.c_float))

    rbf_lib.newWeights.argtypes = [ct.POINTER(ct.POINTER(ct.c_float)), ct.c_int32, ct.c_int32,
                                   ct.POINTER(ct.POINTER(ct.c_float)), ct.c_int32, ct.c_int32,
                                   ct.c_int32, ct.c_int32]
    rbf_lib.newWeights.restype = ct.POINTER(ct.POINTER(ct.c_float))

    rbf_lib.trainingClassification.argtypes = [ct.POINTER(ct.POINTER(ct.c_float)), ct.c_int32, ct.c_int32,
                                               ct.POINTER(ct.POINTER(ct.c_float)), ct.c_int32, ct.c_int32,
                                               ct.POINTER(ct.POINTER(ct.c_float)), ct.c_int32, ct.c_int32,
                                               ct.POINTER(ct.POINTER(ct.c_float)), ct.c_int32, ct.c_int32,
                                               ct.c_int32, ct.c_int32]
    rbf_lib.trainingClassification.restype = ct.POINTER(ct.POINTER(ct.c_float))

    return rbf_lib


# Convert img to an array into an array
def add_img(x: Any, y: Any, file_path: globals(), value: int) -> None:
    for filename in os.listdir(file_path):
        f = os.path.join(file_path, filename)
        if os.path.isfile(f):
            img = PIL.Image.open(f)
            img = img.convert('L')
            img = np.array(img)
            img = np.ravel(img)
            x.append(img)
            y.append(value)


# To convert to a double pointer
def convert_to_pointer(c_type: ct, array_input: Any, num_rows: int, num_cols_in_rows: int) -> Any:
    POINTER_C_TYPE = ct.POINTER(c_type)
    ITLARR = c_type * num_cols_in_rows
    PITLARR = POINTER_C_TYPE * num_rows
    ptr = PITLARR()
    for i in range(num_rows):
        ptr[i] = ITLARR()
        for j in range(num_cols_in_rows):
            ptr[i][j] = array_input[i][j]
    return ptr


def convert_array(array_input: Any) -> Any:
    array_output = convert_to_pointer(ct.c_float, array_input, len(array_input), len(array_input[0]))
    array_output = ct.cast(array_output, ct.POINTER(ct.POINTER(ct.c_float)))
    return array_output


# Save
def save(rbf: ct.cdll, weights: Any, n_rows_weights: int, n_cols_weight: int, str_file: str, percentage: float) -> None:
    str_file = str_file.encode('UTF-8')
    rbf.saveModelRBF(weights, str_file, n_rows_weights, n_cols_weight, percentage)
    rbf.destroyFloatArray(weights, n_rows_weights)


# Testing
def testing(rbf: ct.CDLL, file: str,
            x_testing: Any,
            n_images: int, n_dataset: int,
            gamma: int, mode: int) -> int:
    valid = 0
    binary_file = file.encode('ascii')
    centers_testing = convert_array(x_testing)
    weights = rbf.loadModelRBF(binary_file, n_dataset)

    outputs = [[1, -1, -1, -1],
               [-1, 1, -1, -1],
               [-1, -1, 1, -1],
               [-1, -1, -1, 1]]

    answers = {0: "Tour Eiffel",
               1: "Arc de Triomphe",
               2: "Louvre",
               3: "Pantheon",
               4: "Pas trouvé"}

    print("--- Testing ---")
    res = rbf.newWeights(weights, n_images, n_dataset,
                         centers_testing, len(x_testing), len(x_testing[0]),
                         gamma, mode)

    for dataset in range(0, n_dataset):
        for i in range(0, len(outputs)):
            for j in range(0, n_dataset):
                if int(res[dataset][j]) == outputs[i][j]:
                    valid += 1
                    if valid == 4:
                        print(answers.get(i))
                else:
                    valid = 0
    if valid < 4:
        print(answers.get(4))


# Training
def training_classification(rbf: ct.CDLL,
                            x: Any, y: Any,
                            gamma: int, n_iter: int) -> tuple[Any, int, int]:
    centers = convert_array(x)
    inputs = convert_array(y)
    weights = rbf.initModelWeights(len(y), len(y[0]))

    print(x)
    print(np.shape(x)[0], np.shape(x)[1])

    print("--- TRAINING---")
    w_trained = rbf.newWeights(centers, len(x), len(x[0]),
                               centers, len(x), len(x[0]),
                               inputs, len(y), len(y[0]),
                               weights, len(y), len(y[0]),
                               gamma, n_iter)

    return w_trained, len(y), len(y[0])


if __name__ == '__main__':
    print("--- Starting now ---")

    RBF_SAVE = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/Save/RBF/Classification"

    DS_TYPE = "_32_classique_min"
    DS_TRAIN = ["EIFFEL_TOWER_TRAINING", "TRIUMPHAL_ARC_TRAINING", "LOUVRE_TRAINING", "PANTHEON_TRAINING"]
    DS_TEST = ["EIFFEL_TOWER_TESTING", "TRIUMPHAL_ARC_TESTING", "LOUVRE_TESTING", "PANTHEON_TESTING"]
    OUT = [[1, -1, -1, -1],
           [-1, 1, -1, -1],
           [-1, -1, 1, -1],
           [-1, -1, -1, 1]]

    # Retrieve img
    x_training = []
    y_training = []
    x_testing = []
    y_testing = []
    x_test = []
    y_test = []

    for i in range(len(DS_TRAIN)):
        print(f"Set {DS_TYPE}_{DS_TRAIN[i]} loaded")
        add_img(x_training, y_training, globals()[DS_TRAIN[i] + DS_TYPE], OUT[i])
        print(f"Set {DS_TYPE}_{DS_TEST[i]} loaded")
        add_img(x_testing, y_testing, globals()[DS_TEST[i] + DS_TYPE], OUT[i])

    x_TRAINING = np.array(x_training)
    y_TRAINING = np.array(y_training)
    x_testing = np.array(x_testing)
    y_testing = np.array(y_testing)

    rbf_lib = ct.CDLL(RBF_LIB_PATH)
    rbf = init_dll(rbf_lib)

    n_iter = 100
    gamma = 2

    x = [1, 2,
         2, 3,
         3, 3,
         2.5, 3]
    y = [[1],
         [-1],
         [-1],
         [1]]

    """img_to_test = []
    bin = []
    add_img(img_to_test,
            bin,
            "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/Dataset/DS complet_v2/Test",
            0)"""

    weights, n_rows_weights, n_cols_weights, res = training_classification(rbf,
                                                                           x_training, y_training,
                                                                           gamma, n_iter)
    # save(rbf, weights, n_rows_weights, n_cols_weights,
    #     f"{RBF_SAVE}/{DS_TYPE}_num_iter{n_iter}", res)
    """testing(rbf,
            "C:\\Users\\juanm\\OneDrive\\Bureau\\ESGI - Projets\\3IABD\\Projet Annuel\\Save\\RBF\\Classification\\test.txt",
            img_to_test,
            1, 4,
            2, 1)
    
    
        weights, n_rows_weights, n_cols_weights, res = training_classification(rbf,
                                                                               x, y,
                                                                               gamma, n_iter)
"""