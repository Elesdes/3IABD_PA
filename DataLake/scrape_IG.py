from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
import time
import os
import wget

def scrap_insta_loc(user_name, password,tag_loc, directorie_outpute,file_link):
    #specify the path to chromedriver.exe (download and save on your computer)
    options = webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches", ["enable-logging"])

    driver = webdriver.Chrome(options=options)
    #open the webpage
    driver.get("http://www.instagram.com")

    alert = WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.XPATH, '//button[contains(text(), "Uniquement autoriser les cookies essentiels")]'))).click()

    #target username
    username = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[name='username']")))
    psw = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[name='password']")))

    #enter username and password
    username.clear()
    username.send_keys(user_name)
    psw.clear()
    psw.send_keys(password)

    #target the login button and click it

    button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='submit']"))).click()

    time.sleep(5)
    alert = WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.XPATH, '//button[contains(text(), "Plus tard")]'))).click()
    alert = WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.XPATH, '//button[contains(text(), "Plus tard")]'))).click()

    # target the search input field
    searchbox = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//input[@placeholder='Rechercher']")))
    searchbox.clear()

    # search for the hashtag cat
    keyword = tag_loc
    searchbox.send_keys(keyword)

    # FIXING THE DOUBLE ENTER
    time.sleep(5)  # Wait for 5 seconds
    my_link = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//a[contains(@href, '/" + keyword + "/')]")))
    my_link.click()

    def test2(driver):
        anchors = driver.find_elements_by_tag_name('img')
        # print("anchors1 :", anchors)
        anchors = [a.get_attribute('src') for a in anchors]
        # print("anchors2 :", anchors)
        anchors = [a for a in anchors if str(a).startswith("https://scontent")]
        # anchors = [a for a in anchors if str(a).startswith("https://instagram.fcdg")]
        # print("anchors3 :", anchors)
        return anchors


    #increase the range to sroll more
    time.sleep(30)

    n_scrolls = 20
    all_anchors = []
    for j in range(0, n_scrolls):
        time.sleep(5)
        # driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        driver.execute_script("window.scrollTo({  top: document.body.scrollHeight,  left: 0,  behavior: 'smooth'});")
        time.sleep(5)
        anchors = test2(driver)
        all_anchors.extend(anchors)
        print(j, len(list(set(all_anchors))))

    #target all the link elements on the page

    all_anchors = list(set(all_anchors))
    print('Found ' + str(len(all_anchors)) + ' links to images')
    # txt_all_anchors = '\n'.join([str(item) for item in all_anchors])
    link = []
    link2 = []
    all_anchors2 = []

    if os.stat(file_link).st_size != 0:
        with open(file_link, "r") as fichier1:
            for ligne in fichier1:
                link.append(ligne[:len(ligne)-1])
                link2.append(ligne[:len(ligne)-1])
    print(all_anchors)
    print(link)
    for p in all_anchors:
        if p not in link:
            link2.append(p)
            all_anchors2.append(p)

    with open(file_link, "w") as fichier1:
        for lin in link2:
            fichier1.write(lin+'\n')

    print('Found ' + str(len(all_anchors2)) + ' new links to images')

    path = os.getcwd()
    path = os.path.join(path, directorie_outpute)
    #create the directory
    if not os.path.exists(path):
        os.mkdir(path)
    counter = len(link)
    for image in all_anchors2:
        if (image.find('jpg') != -1) or (image.find('jpeg') != -1):
            ext = ".jpg"
        elif (image.find('png') != -1):
            ext = ".png"

        save_as = os.path.join(path, "img" + str(counter) + ext)
        # print(image, '\n' ,save_as)
        wget.download(image, save_as)
        counter += 1
