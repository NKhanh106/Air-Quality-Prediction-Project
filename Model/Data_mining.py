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
from Data_preprocess import *
base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, "../Data/FinalData.csv")
aqi_path = os.path.join(base_dir, "../Data/AQI.csv")

old_df = pd.read_csv(csv_path)

def air_quality_crawl():
    download_dir = os.path.abspath(os.path.join(base_dir, "../Data"))
    target_path = os.path.join(base_dir, "../Data/hanoi-air-quality.csv")

    # Cấu hình Chrome để lưu file CSV vào thư mục download_dir
    options = webdriver.ChromeOptions()
    prefs = {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    options.add_experimental_option("prefs", prefs)

    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.binary_location = "/usr/bin/chromium"
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
    # Set up options để giảm tài nguyên sử dụng
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-infobars')
    options.add_argument('--disable-notifications')
    options.add_argument('--blink-settings=imagesEnabled=false')
    options.add_argument("--window-size=1920,1080")
    options.binary_location = "/usr/bin/chromium"
    
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

    if not pd.api.types.is_datetime64_any_dtype(old_df['Date']):
        old_df['Date'] = pd.to_datetime(old_df['Date'])
    date = old_df['Date'].max() + datetime.timedelta(days=1)  # Ngày bắt đầu từ ngày tiếp theo của ngày cuối cùng trong dữ liệu cũ
    end_date = datetime.datetime.now() - datetime.timedelta(days=1)  # Ngày kết thúc là hôm qua

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
    # lấy lại phần tử sau khi trang đã reload
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
    try:
        print(f"Đang crawl dữ liệu từ {name}")
        df = scrape_weather_to_df(name)
        return df, name, True
    except Exception as e:
        return pd.DataFrame(), name, False

def weather_data_crawl():
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
    df = weather_data_crawl()
    df = process_weather_data(df)
    
    aqi_df = air_quality_crawl()
    aqi_df = process_AQI_data(aqi_df)
    aqi_df = aqi_df[(aqi_df['Date'] > old_df['Date'].max()) & (aqi_df['Date'] < datetime.datetime.now())]

    df = pd.merge(aqi_df, df, on='Date', how='inner')

    final_df = pd.concat([old_df, df], ignore_index=True)
    final_df.drop(columns=['Day of Year'], inplace=True)
    final_df.to_csv(csv_path, index=False)