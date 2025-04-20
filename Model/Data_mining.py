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

def scrape_weather_to_df(name, location_id):
    # Set up Chrome options to optimize resource usage
    options = webdriver.ChromeOptions()
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-infobars')
    options.add_argument('--disable-notifications')
    options.add_argument('--blink-settings=imagesEnabled=false')  # Disable image loading

    driver = webdriver.Chrome(options=options)
    url = f"https://www.weatherapi.com/history/q/ha-noi-2717933?loc=2717933"
    driver.get(url)

    data = []
    try:
        # Handle cookie consent popup if it exists
        allow_cookies = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "accept-cookies"))  # Adjust ID based on actual website
        )
        allow_cookies.click()
        time.sleep(1)
    except (NoSuchElementException, TimeoutException):
        pass

    # Define the metrics to scrape based on the image
    record_keys = ['Time', 'Temp', 'Wind', 'Precip', 'Cloud', 'Humidity', 'Pressure']
    date = datetime.datetime(2015, 1, 1)
    end_date = datetime.datetime(2025, 4, 17)

    try:
        while date < end_date:
            date_str = date.strftime('%Y-%m-%d')
            try:
                # Locate the date input field and submit the date
                input_date = WebDriverWait(driver, 7).until(
                    EC.presence_of_element_located((By.ID, 'ctl00_MainContentHolder_txtpastdate'))
                )

                # Nếu cần trigger event để cập nhật dữ liệu theo ngày:
                driver.execute_script("arguments[0].dispatchEvent(new Event('change'))", input_date)

                # Nếu trang cần click nút "Tìm kiếm" sau khi chọn ngày:
                search_btn = driver.find_element(By.ID, 'ctl00_MainContentHolder_butHistory')
                search_btn.click()

            except WebDriverException:
                date += datetime.timedelta(days=1)
                continue

            time.sleep(2)  # Wait for the page to load

            # Locate the weather data (assuming it's in a table or div structure)
            try:
                weather_container = driver.find_element(By.XPATH, '/html/body/form/div[4]/section/div[2]/section/main/div/div[2]/div[1]/div/div/div/table[2]/tbody[2]') 
                rows = weather_container.find_elements(By.CLASS_NAME, "table-dark")

                for row in rows:
                    try:
                        # Extract the required metrics
                        temp = row.find_element(By.CLASS_NAME, "Temp").text.strip()
                        wind = row.find_element(By.CLASS_NAME, "Wind").text.strip()
                        precip = row.find_element(By.CLASS_NAME, "Precip").text.strip()
                        cloud = row.find_element(By.CLASS_NAME, "Cloud").text.strip()
                        humidity = row.find_element(By.CLASS_NAME, "Humidity").text.strip()
                        pressure = row.find_element(By.CLASS_NAME, "Pressure").text.strip()

                        values = [time, temp, wind, precip, cloud, humidity, pressure]

                        if values:
                            data.append([date_str] + values)
                    except Exception:
                        continue
            except Exception:
                date += datetime.timedelta(days=1)
                continue

            date += datetime.timedelta(days=1)
    finally:
        driver.quit()

    if data:
        df = pd.DataFrame(data, columns=["Date"] + record_keys)
        return df
    else:
        return pd.DataFrame(columns=["Date"] + record_keys)

def process_location(name, location_id, output_dir):
    try:
        print(f"Đang crawl dữ liệu từ {name}")
        df = scrape_weather_to_df(name, location_id)
        filename = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(filename, index=False)
        return f"✅ Đã lưu dữ liệu {name} vào {filename}", name, True
    except Exception as e:
        return f"❌ Lỗi khi crawl {name}: {e}", name, False

def main():
    output_dir = r"C:\Users\namkh\Downloads"
    os.makedirs(output_dir, exist_ok=True)

    # Location details (Hanoi with the given location ID)
    locations = [("ha-noi", "2717933")]

    # Number of concurrent threads (keep it low to avoid being blocked)
    max_workers = 1

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        future_to_location = {
            executor.submit(process_location, name, loc_id, output_dir): name
            for name, loc_id in locations
        }

        # Show progress
        with tqdm(total=len(locations), desc="🌤️ Crawling weather data") as pbar:
            for future in concurrent.futures.as_completed(future_to_location):
                name = future_to_location[future]
                try:
                    message, location, success = future.result()
                    print(message)
                    results.append((location, success))
                except Exception as e:
                    print(f"❌ Lỗi với {name}: {e}")
                    results.append((name, False))
                pbar.update(1)

    # Summarize results
    success_count = sum(1 for _, success in results if success)
    print(f"\n✅ Đã crawl thành công: {success_count}/{len(locations)}")

    if success_count < len(locations):
        failed_locations = [name for name, success in results if not success]
        print(f"❌ Các địa điểm chưa crawl được: {', '.join(failed_locations)}")

if __name__ == "__main__":
    main()