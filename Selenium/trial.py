import selenium
from selenium import webdriver

# Create a WebDriver object
driver = webdriver.Chrome()

# Navigate to the admin page
driver.get("https://www.example.com/admin")

# Find the buttons that you want to click
button_1 = driver.find_element_by_id("button_1")
button_2 = driver.find_element_by_id("button_2")

# Click on the buttons
button_1.click()
button_2.click()

# Verify that the content of the page has changed
assert driver.find_element_by_id("content_1").text == "This is the content for button 1."
assert driver.find_element_by_id("content_2").text == "This is the content for button 2."

# Close the browser
driver.quit()
