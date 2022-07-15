import os

if __name__ == '__main__':
    path = "D:/PA/models_save/LINEAR_SAVE/a_recup"
    list_dosier = [ f for f in os.listdir(path) if not os.path.isfile(os.path.join(path,f)) ]
    print(list_dosier)
    for doss in list_dosier:
        path_doss = f"{path}/{doss}"
        list_dosier2 = [f for f in os.listdir(path_doss) if not os.path.isfile(os.path.join(path_doss, f))]
        résultat_test
        for num in list_dosier2:
            path_num = f"{path_doss}/{num}"
            list_dosier3 = [f for f in os.listdir(path_num)]

            print(list_dosier3)