import streamlit as st
import pandas as pd
import os
import datetime
import sys
base_path = os.path.dirname(os.path.abspath(__file__))
utils_path = os.path.join(base_path, './Model')
sys.path.append(utils_path)

from Main import predict, call_chart
from Data_mining import update_weather_data

chart_inform = {
    "category" : "co",
    "start_date" : datetime.datetime.now() - datetime.timedelta(days=1),
    "end_date" : datetime.datetime.now() - datetime.timedelta(days=1)
}

para_unit = {
    "co" : "mg/m³",
    "pm10" : "µg/m³",
    "pm25" : "µg/m³",
    "no2" : "µg/m³",
    "so2" : "µg/m³",
    "o3" : "µg/m³",
    "Temp" : '°c',
    "Wind" : 'km/h',
    "Gust" : 'km/h',
    "Pressure" : 'mb',
    "Cloud" : '%',
    "Rain" : '\nmm'
}

st.sidebar.title("Thông tin dự án")
st.sidebar.header("Tên dự án: Dự đoán chất lượng không khí thành phố Hà Nội")
st.sidebar.write("Người thực hiện: Chu Nam Khánh")

tab1, tab2 = st.tabs(["Dự báo chất lượng không khí hôm nay", "Biểu đồ thông số không khí"])

with tab1:
    st.header("Dự báo chất lượng không khí thành phố Hà Nội ngày hôm nay.")
    st.subheader("Nhấn nút 'Cập nhật' để hệ thống cập nhật dữ liệu.")

    if st.button("Cập nhật"):
        #update_weather_data()
        st.write(f"Đã cập nhật dữ liệu đến {(datetime.datetime.now() - datetime.timedelta(days=1)).date()}.")

    st.divider()

    st.subheader(f"Nhấn nút dự đoán để đưa ra dự đoán chất lượng không khí của mô hình cho {(datetime.datetime.now()).date()}.")
    if st.button("Dự đoán"):
        predict_parameter, aqi, main_pollutant, attention = predict()
        st.subheader("Dự đoán của mô hình là :")
        for key in predict_parameter:
            st.write(f"Chỉ số {key} : {predict_parameter[key]:.3f} {para_unit[key]}")

        st.write(f"Chỉ số chất lượng không khí(AQI) dự đoán là : {aqi}")
        st.write(f"Chất gây ô nhiễm chính được dự đoán là : {main_pollutant}")
        st.write("!!!", attention)
        

with tab2:
    st.header("Quan sát biểu đồ của các thông số không khí.")

    with st.form(key = "chart information"):
        chart_inform['category'] = st.selectbox("Thông số :", ["co", "pm10", "pm25", "o3", "no2", "so2", "Temp", "Rain", "Cloud", "Pressure", "Wind", "Gust"])
        chart_inform["start_date"] = st.date_input("Nhập ngày bắt đầu :")
        chart_inform["end_date"] = st.date_input("Nhập ngày kết thúc :")

        submit_button = st.form_submit_button(label= "Submit")

        if submit_button:
            st.write(f"Biểu đồ của {chart_inform['category']} từ ngày  {chart_inform["start_date"]}  đến ngày  {chart_inform["end_date"]}  :")
            df_chart = call_chart(chart_inform["category"], chart_inform["start_date"], chart_inform["end_date"])
            df_chart.set_index("Date", inplace=True)
            st.line_chart(df_chart)


