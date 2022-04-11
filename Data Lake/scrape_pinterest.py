import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import time


def web_scraping_Pinterest(url):
    ScrollNumber = 65
    sleepTimer = 2

    options = webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches", ["enable-logging"])

    driver = webdriver.Chrome(options=options)
    driver.get(url)
    i=0
    for _ in range(1, ScrollNumber):
        i+=1
        driver.execute_script("window.scrollTo(1,100000)")
        print("scrolling",i)
        time.sleep(sleepTimer)

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    i = 0
    for link in soup.find_all('img'):
        print(i)
        i += 1
        url = 'EiffelTower_picture/Pinterest/img' + str(i) + "." + link.get('src')[len(link.get('src')) - 3:]
        img_data = requests.get(link.get('src')).content
        with open(url, 'wb') as handler:
            handler.write(img_data)
