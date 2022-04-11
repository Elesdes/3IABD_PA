#imports here
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
import time
import os
import wget

#specify the path to chromedriver.exe (download and save on your computer)
options = webdriver.ChromeOptions()
options.add_experimental_option("excludeSwitches", ["enable-logging"])

driver = webdriver.Chrome(options=options)
#open the webpage
driver.get("http://www.instagram.com")

alert = WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.XPATH, '//button[contains(text(), "Uniquement autoriser les cookies essentiels")]'))).click()

#target username
username = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[name='username']")))
password = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[name='password']")))

#enter username and password
username.clear()
username.send_keys("")
password.clear()
password.send_keys("")

#target the login button and click it

button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='submit']"))).click()

#We are logged in!

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

time.sleep(5)
alert = WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.XPATH, '//button[contains(text(), "Plus tard")]'))).click()
alert = WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.XPATH, '//button[contains(text(), "Plus tard")]'))).click()

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# target the search input field
searchbox = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//input[@placeholder='Rechercher']")))
searchbox.clear()

# search for the hashtag cat
keyword = "tour-eiffel"
searchbox.send_keys(keyword)

# FIXING THE DOUBLE ENTER
time.sleep(5)  # Wait for 5 seconds
my_link = WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable((By.XPATH, "//a[contains(@href, '/" + keyword + "/')]")))
my_link.click()

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def test(driver):
    anchors = driver.find_elements_by_tag_name('a')
    anchors = [a.get_attribute('href') for a in anchors]
    anchors = [a for a in anchors if str(a).startswith("https://www.instagram.com/p/")]
    # print(driver.page_source)
    return anchors

#scroll down 2 times
#increase the range to sroll more
time.sleep(5)

n_scrolls = 1
all_anchors = []
for j in range(0, n_scrolls):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(5)
    anchors = test(driver)
    all_anchors.extend(anchors)
    print(j)

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#target all the link elements on the page
print('Found ' + str(len(all_anchors)) + ' links to images')
all_anchors = list(set(all_anchors))
print('Found ' + str(len(all_anchors)) + ' links to images')
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

images = []

# follow each image link and extract only image at index=1
for a in all_anchors:
    driver.get(a)
    time.sleep(2)
    img = driver.find_elements_by_tag_name('img')
    img = [i.get_attribute('src') for i in img]
    images.append(img[0])



# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


path = os.getcwd()
path = os.path.join(path, "EiffelTower_picture\Insta")

#create the directory
os.mkdir(path)
counter = 0
for image in images:
    save_as = os.path.join(path, "img" + str(counter) + '.jpg')
    wget.download(image, save_as)
    counter += 1

























# from instascrape import *
# # Instantiate the scraper objects
# google = Profile('https://www.instagram.com/google/')
# google_post = Post('https://www.instagram.com/p/CG0UU3ylXnv/')
# google_hashtag = Hashtag('https://www.instagram.com/explore/tags/google/')
#
# google.scrape()
# google_post.scrape()
# google_hashtag.scrape()
#
# print(google.followers)
# print(google_post['hashtags'])
# test = google_hashtag.get_recent_posts(amt = 75)
# print(test)