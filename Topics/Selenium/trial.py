from selenium import webdriver

# Create a new instance of the Chrome driver
driver = webdriver.Chrome()

# Navigate to the Google homepage
driver.get("https://www.google.com")

# Find the search input box
search_box = driver.find_element_by_name("q")

# Enter the search term
search_box.send_keys("Selenium")

# Click the search button
search_box.submit()

# Wait for the search results to load
driver.implicitly_wait(10)

# Print the title of the first search result
print(driver.find_element_by_xpath("//h3[@class='LC20lb']/a").text)

# Close the browser
driver.quit()
