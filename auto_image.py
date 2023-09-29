from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import numpy as np


ras = np.arange(5,180,5)
decs = np.arange(5,85,5)

# Initialize Selenium webdriver
driver = webdriver.Chrome(executable_path='C:/Users/twbb2/Documents/chromedriver.exe')

# Navigate to the page
driver.get("https://archive.stsci.edu/cgi-bin/dss_form")

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

# Add explicit wait for the checkbox
wait = WebDriverWait(driver, 10)
checkbox = wait.until(EC.presence_of_element_located((By.NAME, 's')))

# Click to check the checkbox
checkbox.click()

for rax in ras:
    for decx in decs:
        print('RA ', rax)
        print('DEC ', decx)

        # Find elements and fill in the form
        ra = driver.find_element_by_name("r")
        ra.clear()
        ra.send_keys(str(rax)+"d")

        dec = driver.find_element_by_name("d")
        dec.clear()
        dec.send_keys(str(decx)+"d")

        # Submit the form
        wait = WebDriverWait(driver, 10)
        element = wait.until(EC.presence_of_element_located((By.XPATH, "/html/body/b/form[2]/center[2]/input[1]")))
        element.click()

        time.sleep(2)

# Close the browser
driver.close()
