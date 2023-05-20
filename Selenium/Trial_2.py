# import webdriver
from selenium import webdriver

# import Action chains
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys

# create webdriver object
driver = webdriver.Chrome("/Users/apple/Downloads/chromedriver_mac_arm64/chromedriver")

driver.maximize_window() # For maximizing window

# get geeksforgeeks.org
driver.get("https://www.geeksforgeeks.org/")

driver.maximize_window() # For maximizing window

driver.implicitly_wait(5) 

search = driver.find_element_by_xpath("/html/body/div[3]/div/div/div/div[1]/div[1]/div[2]/span/span/span[1]/input")
search.send_keys("Python")
search.send_keys(Keys.RETURN)


