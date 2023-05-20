# Python program to demonstrate
# selenium

# import webdriver
from selenium import webdriver

# create webdriver object

driver = webdriver.Chrome("/Users/apple/Downloads/chromedriver_mac_arm64/chromedriver")

# get google.co.in
driver.get("https://google.com/search?q=geeksforgeeks")
