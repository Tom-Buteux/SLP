

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.add_argument("--headless")




# Initialize Selenium webdriver
driver = webdriver.Chrome(executable_path='C:/Users/twbb2/Documents/chromedriver.exe')

# Navigate to the page
driver.get("https://archive.stsci.edu/cgi-bin/dss_form")

# Find elements and fill in the form
ra = driver.find_element_by_name("r")
ra.clear()
ra.send_keys("10")

dec = driver.find_element_by_name("d")
dec.clear()
dec.send_keys("+10")

height = driver.find_element_by_name("h")
height.clear()
height.send_keys("60")

width = driver.find_element_by_name("w")
width.clear()
width.send_keys("60")
# You can also select options from dropdowns, click checkboxes, etc.
select = Select(driver.find_element_by_name('f'))
select.select_by_value('gif')

select = Select(driver.find_element_by_name('v'))
select.select_by_value('quickv')


# Find the checkbox with name='s' and click it to check it
checkbox = driver.find_element_by_name('s')
checkbox.click()


#submit the form

wait = WebDriverWait(driver, 10)
element = wait.until(EC.presence_of_element_located((By.XPATH, "/html/body/b/form[2]/center[2]/input[1]")))
element.click()

# Close the browser
driver.close()
