import os

import requests
import wget
from bs4 import BeautifulSoup
from selenium import webdriver
import time


def web_scraping_Pinterest(url):
    ScrollNumber = 100
    sleepTimer = 2

    options = webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches", ["enable-logging"])

    driver = webdriver.Chrome(options=options)
    driver.get(url)
    i=0
    for _ in range(1, ScrollNumber):
        i+=1
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
        print("scrolling",i)
        time.sleep(sleepTimer)

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    links = []

    for link in soup.find_all('img'):
        links.append(link.get('src'))
    print('Found ' + str(len(links)) + ' links to images')

    link = []
    link2 = []
    links2 = []

    if os.stat("link.txt").st_size != 0:
        with open("link.txt", "r") as fichier1:
            for ligne in fichier1:
                link.append(ligne[:len(ligne) - 1])
                link2.append(ligne[:len(ligne) - 1])

    for p in links:
        if p not in link:
            link2.append(p)
            links2.append(p)

    with open("link.txt", "w") as fichier1:
        for lin in link2:
            fichier1.write(lin + '\n')

    print('Found ' + str(len(links2)) + ' new links to images')





    path = os.getcwd()
    path = os.path.join(path, "EiffelTower_picture\Pinterest")
    if not os.path.exists(path):
        os.mkdir(path)
    counter = len(link)
    for image in links2:
        ext = "." + image[-3:]

        save_as = os.path.join(path, "img" + str(counter) + ext)
        wget.download(image, save_as)
        counter += 1