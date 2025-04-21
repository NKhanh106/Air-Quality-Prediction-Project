import pandas as pd
import numpy as np
import os

# Thêm cột "Day of Year" (0: thứ Hai, ..., 6: Chủ Nhật)
base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, "../Data/FinalData.csv")

main_data = pd.read_csv(csv_path)
main_data['Date'] = pd.to_datetime(main_data['Date'], errors='coerce')
main_data['Day of Year'] = main_data['Date'].dt.dayofyear


# Hàm tính giá trị trung bình cho cùng ngày trong tuần
def calculate_avg(day_of_week, column):
    filtered_data = main_data[(main_data['Day of Year'] == day_of_week) & (main_data[column] != 0)]
    return filtered_data[column].mean() if len(filtered_data) > 0 else np.nan


def process_AQI_data(source_df):
    #Ngoại lệ do sai lệch trong quá trình đo đạc
    source_df.loc[(source_df['o3'] >= 200), 'o3'] = 0.0
    source_df.loc[(source_df['no2'] >= 120), 'no2'] = 0.0

    #Định dạng lại các kiểu dữ liệu trong các cột
    source_df['Date'] = source_df['Date'].astype('datetime64[ns]')
    source_df['co'] = source_df['co'].astype(float)
    source_df['no2'] = source_df['no2'].astype(float)
    source_df['o3'] = source_df['o3'].astype(float)
    source_df['pm10'] = source_df['pm10'].astype(float)
    source_df['pm25'] = source_df['pm25'].astype(float)
    source_df['so2'] = source_df['so2'].astype(float)

    #Tính các giá trị tương ứng và trung bình các điểm đo được trong các điểm đo khác nhau
    for feature in ['co', 'no2', 'o3', 'pm10', 'pm25', 'so2']:
        for i in range(len(source_df)):
            time = source_df.loc[i, 'Date']
            values = []

            match = source_df[source_df['Date'] == time]
            if not match.empty:
                val = match[feature].values[0]
                if val != 0:  # Bỏ qua các giá trị 0
                    values.append(val)

            if values:
                source_df.loc[i, feature] = round(sum(values) / len(values), 2)

    source_df['Day of Year'] = source_df['Date'].dt.dayofyear

    # Thay thế giá trị 0 bằng giá trị trung bình và một chút nhiễu động ngẫu nhiên
    for index, row in source_df.iterrows():
        for column in ['co', 'no2', 'o3', 'pm10', 'pm25', 'so2']:
            if row[column] == 0:
                avg_value = calculate_avg(row['Day of Year'], column)
                source_df.at[index, column] = round(avg_value * (1 + np.random.uniform(-0.5, 0.5)),2)

    return source_df
                

def process_weather_data(source_df):
    #Bỏ cột không cần thiết
    source_df.drop(columns=['Weather'], inplace=True)

    #Định dạng lại các kiểu dữ liệu trong các cột
    source_df['Date'] = source_df['Date'].astype('datetime64[ns]')
    source_df['Temp'] = source_df['Temp'].str.replace('°c', '').str.strip()
    source_df['Rain'] = source_df['Rain'].str.replace('\nmm', '').str.strip()
    source_df['Cloud'] = source_df['Cloud'].str.replace('%', '').str.strip()
    source_df['Pressure'] = source_df['Pressure'].str.replace('mb', '').str.strip()
    source_df['Wind'] = source_df['Wind'].str.replace('km/h', '').str.strip()
    source_df['Gust'] = source_df['Gust'].str.replace('km/h', '').str.strip()

    source_df = source_df.astype({
        'Temp': 'float64',
        'Rain': 'float64',
        'Cloud': 'float64',
        'Pressure': 'float64',
        'Wind': 'float64',
        'Gust': 'float64'
    })

    source_df = source_df.groupby('Date')[['Temp', 'Rain', 'Cloud', 'Pressure', 'Wind', 'Gust']].mean().round(2).reset_index()

    return source_df