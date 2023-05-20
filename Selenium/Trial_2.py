import time
from selenium import webdriver

PATH = "/Users/apple/Downloads/chromedriver_mac_arm64/chromedriver"
driver = webdriver.Chrome(PATH)

# Instruct the WebDriver object to open the URL of your localhost application
driver.get("http://www.google.com")

# Enter your username and password in the login form
username_input = driver.find_element_by_id("Gmail")
username_input.send_keys("your_username")

password_input = driver.find_element_by_id("Images")
password_input.send_keys("your_password")

# Click on the login button
login_button = driver.find_element_by_id("login_button")
login_button.click()

# Wait for the login to be successful
time.sleep(5)

# Verify that you are logged in by checking the page title
assert driver.title == "Your localhost application"
