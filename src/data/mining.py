"""
Data Mining Module - Web scraping cho AQI và weather data
"""

import concurrent.futures
from tqdm import tqdm
import os
import pandas as pd
import time
import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException

from ..utils.paths import get_data_path, ensure_dirs, DATA_EXTERNAL
from .preprocessing import process_AQI_data, process_weather_data

# Ensure directories exist
ensure_dirs()

def air_quality_crawl():
    """Crawl dữ liệu AQI từ aqicn.org"""
    ensure_dirs()  # Ensure directories exist
    download_dir = DATA_EXTERNAL
    target_path = DATA_EXTERNAL / "hanoi-air-quality.csv"
    
    # Cấu hình Chrome để lưu file CSV
    options = webdriver.ChromeOptions()
    prefs = {
        "download.default_directory": str(download_dir.resolve()),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    options.add_experimental_option("prefs", prefs)
    options.add_argument("--start-maximized")
    
    driver = webdriver.Chrome(options=options)
    driver.get("https://aqicn.org/historical/vn/#!city:vietnam/hanoi")

    wait = WebDriverWait(driver, 15)

    click1_xpath = "/html/body/div[7]/div[1]/div[1]/div[5]/div[2]/div/center[1]/span[2]/div"
    click1 = wait.until(EC.element_to_be_clickable((By.XPATH, click1_xpath)))
    driver.execute_script("arguments[0].scrollIntoView(true);", click1)
    time.sleep(1)
    driver.execute_script("arguments[0].click();", click1)

    new_element_xpath = "/html/body/div[7]/div[1]/div[1]/div[5]/div[2]/div/center[1]/span[2]/div/center/div"
    wait.until(EC.presence_of_element_located((By.XPATH, new_element_xpath)))

    if os.path.exists(target_path):
        os.remove(target_path)

    click2 = driver.find_element(By.XPATH, new_element_xpath)
    driver.execute_script("arguments[0].scrollIntoView(true);", click2)
    time.sleep(1)
    driver.execute_script("arguments[0].click();", click2)
    time.sleep(5)
    driver.quit()
    
    df = pd.read_csv(target_path)
    
    return df

def scrape_weather_to_df(name):
    """Scrape weather data từ worldweatheronline.com"""
    # Load old data để biết ngày bắt đầu
    final_data_path = get_data_path("FinalData.csv", "processed")
    if os.path.exists(final_data_path):
        old_df = pd.read_csv(final_data_path)
    else:
        old_df = pd.DataFrame(columns=['Date'])
    
    # Set up Chrome options
    options = webdriver.ChromeOptions()
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-infobars')
    options.add_argument('--disable-notifications')
    options.add_argument('--blink-settings=imagesEnabled=false')
    
    driver = webdriver.Chrome(options=options)
    url = f"https://www.worldweatheronline.com/{name}-weather-history/vn.aspx"
    driver.get(url)
    
    data = []
    try:
        allow_cookies = WebDriverWait(driver, 15).until(
            EC.element_to_be_clickable((By.ID, "CybotCookiebotDialogBodyButtonAccept"))
        )
        allow_cookies.click()
        time.sleep(1)
    except (NoSuchElementException, TimeoutException):
        pass
    
    record_keys = ['Time', 'Weather', 'Temp', 'Rain', 'Cloud', 'Pressure', 'Wind', 'Gust']

    if not old_df.empty and 'Date' in old_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(old_df['Date']):
            old_df['Date'] = pd.to_datetime(old_df['Date'])
        date = old_df['Date'].max() + datetime.timedelta(days=1)
    else:
        date = datetime.datetime(2015, 1, 1)
    
    end_date = datetime.datetime.now() - datetime.timedelta(days=1)

    try:
        while date < end_date:
            date_str = date.strftime('%Y-%m-%d')
            try:
                input_date = WebDriverWait(driver, 7).until(
                    EC.presence_of_element_located((By.ID, 'ctl00_MainContentHolder_txtPastDate'))
                )
                driver.execute_script("arguments[0].value = arguments[1];", input_date, date_str)
                submit_date = driver.find_element(By.ID, 'ctl00_MainContentHolder_butShowPastWeather')
                submit_date.click()
            except WebDriverException:
                date += datetime.timedelta(days=1)
                continue
            
            time.sleep(1)
            
            tables = driver.find_element(By.XPATH, "/html/body/form/div[3]/section/div/div/div/div[3]/div[1]/div/div[3]/table/tbody")
            all_rows = tables.find_elements(By.TAG_NAME, "tr")
            rows = all_rows[2:10] 
            
            for row in rows:
                try:
                    cells = row.find_elements(By.CLASS_NAME, "days-details-row-item1")
                    rains = row.find_elements(By.CLASS_NAME, "days-rain-number")
                    rain = rains[0].text
                    weather_img = cells[1].find_element(By.TAG_NAME, "img")
                    weather = weather_img.get_attribute("title")
                    
                    values = [cells[0].text.strip(), weather, cells[2].text.strip(), rain, cells[3].text.strip(), 
                            cells[4].text.strip(), cells[5].text.strip(), cells[6].text.strip()]
                    
                    if values:
                        data.append([date_str] + values)
                except Exception:
                    continue
            
            date += datetime.timedelta(days=1)
    finally:
        driver.quit()
    
    if data:
        df = pd.DataFrame(data, columns=["Date"] + record_keys)
        return df
    else:
        return pd.DataFrame(columns=["Date"] + record_keys)

def process_location(name):
    """Process một location"""
    try:
        print(f"Đang crawl dữ liệu từ {name}")
        df = scrape_weather_to_df(name)
        return df, name, True
    except Exception as e:
        return pd.DataFrame(), name, False

def weather_data_crawl():
    """Crawl weather data cho tất cả locations"""
    name_tinh_process = ["ha-noi"]
    max_workers = 1
    
    results = []
    dfs = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_name = {
            executor.submit(process_location, name): name 
            for name in name_tinh_process
        }
        
        with tqdm(total=len(name_tinh_process), desc="🌤️ Crawling weather data") as pbar:
            for future in concurrent.futures.as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    df, location, success = future.result()
                    if success:
                        dfs.append(df)
                        print(f"✅ Đã lấy dữ liệu {location}")
                    else:
                        print(f"❌ Không lấy được dữ liệu từ {location}")
                    results.append((location, success))
                except Exception as e:
                    print(f"❌ Lỗi với {name}: {e}")
                    results.append((name, False))
                pbar.update(1)
    
    final_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    return final_df

def update_weather_data():
    """Cập nhật dữ liệu mới từ web"""
    # Load old data
    final_data_path = get_data_path("FinalData.csv", "processed")
    aqi_path = get_data_path("AQI.csv", "raw")
    
    if os.path.exists(final_data_path):
        old_df = pd.read_csv(final_data_path)
    else:
        old_df = pd.DataFrame(columns=['Date'])
    
    # Crawl weather data
    df = weather_data_crawl()
    df = process_weather_data(df)
    
    # Crawl AQI data
    aqi_df = air_quality_crawl()
    aqi_df = process_AQI_data(aqi_df)
    
    # Filter new AQI data
    if not old_df.empty and 'Date' in old_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(old_df['Date']):
            old_df['Date'] = pd.to_datetime(old_df['Date'])
        aqi_df = aqi_df[(aqi_df['Date'] > old_df['Date'].max()) & (aqi_df['Date'] < datetime.datetime.now())]
    
    # Merge AQI và weather
    df = pd.merge(aqi_df, df, on='Date', how='inner')

    # Combine với old data
    if not old_df.empty:
        final_df = pd.concat([old_df, df], ignore_index=True)
    else:
        final_df = df
    
    # Remove Day of Year nếu có
    if 'Day of Year' in final_df.columns:
        final_df.drop(columns=['Day of Year'], inplace=True)
    
    # Save
    final_df.to_csv(final_data_path, index=False)
    print(f"✅ Đã lưu dữ liệu tại: {final_data_path}")

