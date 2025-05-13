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

#sidebar
st.sidebar.title("Thông tin dự án")
st.sidebar.header("Tên dự án: **Dự đoán chất lượng không khí thành phố Hà Nội**")
st.sidebar.write("Người thực hiện : **Chu Nam Khánh**")
st.sidebar.write("Mô tả : Dự án này lấy thông tin dữ liệu về thời tiết và các chỉ số không khí của thành phố Hà Nội từ 1-1-2015 đến nay, huấn luyện mô hình và đưa ra dự đoán các chỉ số cho thời gian ngắn tiếp đấy.")
st.sidebar.write("- Mô hình sử dụng trong dự án này là mô hình **LSTM** với điểm mạnh là học được đặc trưng của chuỗi dữ liệu.")
st.sidebar.write("- Bên cạnh đó có sử dụng thêm mô hình **Random Forests** để thử xem với sức mạnh của mô hình học máy thì có thể cho ra kết quả đáng chú ý hay không.")
st.sidebar.write("Mô hình sẽ đưa ra dự đoán các chỉ số cho ngày tiếp theo, từ đó đánh giá được chỉ số AQI, đưa ra được dự đoán và dự báo cho tình trạng không khí của thành phố Hà Nội, cảnh báo được cho người dùng để hạn chế được tác động của ô nhiễm đến cơ thể mình.")
st.sidebar.write("Bên cạnh đó dự án còn cung cấp lựa chọn để người dùng có thể xem được sự biến động của các yếu tố không khí và thời tiết trong khoảng thời gian nhất định, giúp người dùng có cái nhìn tổng quan về sự biến động của các chất theo từng thời điểm trong năm.")
st.sidebar.divider()

tab1, tab2, tab3 = st.tabs(["Dự báo chất lượng không khí hôm nay", "Biểu đồ thông số không khí", "Thông tin về dự án"])

with tab1:
    st.header("Dự báo chất lượng không khí thành phố Hà Nội ngày hôm nay.")
    st.subheader("Nhấn nút 'Cập nhật' để hệ thống cập nhật dữ liệu.")

    if st.button("Cập nhật"):
        update_weather_data()
        st.write(f"Đã cập nhật dữ liệu đến {(datetime.datetime.now() - datetime.timedelta(days=1)).date()}.")

    st.divider()

    st.subheader(f"Nhấn nút dự đoán để đưa ra dự đoán chất lượng không khí của mô hình cho {(datetime.datetime.now()).date()}.")
    if st.button("Dự đoán"):
        predict_parameter, aqi, main_pollutant, attention = predict()
        st.subheader("Dự đoán của mô hình là :")
        for key in predict_parameter:
            st.write(f"- Chỉ số {key} : {predict_parameter[key]:.3f} {para_unit[key]}")

        st.write(f"(+) Chỉ số chất lượng không khí(AQI) dự đoán là : {aqi}")
        st.write(f"(+) Chất gây ô nhiễm chính được dự đoán là : {main_pollutant}")
        st.subheader("Cảnh báo của các chuyên gia :")
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


with tab3:
    st.header("Thông tin về dự án.")

    st.divider()

    st.write("- Dữ liệu của dự án :")
    st.markdown(
        'Dữ liệu về các chỉ số chất lượng không khí lấy từ đường dẫn : [Dữ liệu chất lượng không khí](https://aqicn.org/historical/vn/#!city:vietnam/hanoi)'
    )
    st.markdown(
        'Dữ liệu về thời tiết Hà Nội lấy từ đường dẫn : [Dữ liệu thời tiết](https://www.worldweatheronline.com/ha-noi-weather-history/vn.aspx)'
    )

    st.divider()

    st.write("- Các mô hình sử dụng :")
    st.write("**Mô hình LSTM(Long Short - Term Memmory)** : một mô hình học sâu ứng dụng mạng nơ-ron hồi tiếp(RNN - Recurrent Neural Network), được thiết kể để xử lý và dự đoán những dữ dữ liệu dạng chuỗi giá trị, khi mà dữ liệu hiện tại phụ thuộc vào các dữ liệu trước đó.")
    st.markdown('<a href="https://viblo.asia/p/tim-hieu-lstm-bi-quyet-giu-thong-tin-lau-dai-hieu-qua-MG24BaezVz3" target="_blank">Mô hình LSTM</a>', unsafe_allow_html=True)
    st.write("**Mô hình Random Forests** : là một mô hình học máy mạnh mẽ và phổ biến hiện nay, thuộc loại học có giám sát(supervised learning). Mô hình này bao gồm nhiều cây quyết định con(Decision Tree), mỗi cây sẽ được huấn luyện trên 1 tập dữ liệu con và tại mỗi điểm phân chia(nhánh) thì sẽ chia dữ liệu theo các đặc trưng của chúng. Cuối cùng, mô hình sẽ lấy kết quả đa số(bài toán phân loại) hoặc là giá trị trung bình(bài toán hồi quy) của các cây để đưa ra kết quả của mô hình.")
    st.markdown('<a href="https://viblo.asia/p/phan-lop-bang-random-forests-trong-python-djeZ1D2QKWz" target="_blank">Mô hình RandomForests</a>', unsafe_allow_html=True)

    st.divider()

    st.write("- Chất lượng không khí :")
    st.write("-- Chỉ số **AQI**(Air Quality Index) là một chỉ số tổng hợp được sử dụng để đánh giá chất lượng không khí hàng ngày và ảnh hưởng của không khí đến sức khỏe con người.")
    st.write("-- Các chất gây ô nhiễm chính trong AQI là PM2.5, PM10, O3, NO₂, SO₂, CO,... Với các tác động có hại đến sức khỏe.")
    st.write("(+) PM2.5 : Bụi mịn đường kính < 2.5 µm – nguy hiểm nhất.")
    st.write("(+) PM10 : Bụi có đường kính < 10 µm.")
    st.write("(+) O₃ (Ozone mặt đất) : 	Gây kích ứng mắt, họng, phổi.")
    st.write("(+) NO₂ (Nitrogen Dioxide) : Gây viêm phổi, khó thở.")
    st.write("(+) SO₂ (Sulfur Dioxide) : Gây ho, khó thở, viêm đường hô hấp.")
    st.write("(+) CO (Carbon Monoxide) : Gây thiếu oxy, ngất, nguy hiểm tính mạng.")
    st.write("-- Mỗi chất được tính AQI riêng, và AQI cuối cùng là giá trị AQI cao nhất trong các giá trị AQI. Ví dụ ta có  AQI(PM2.5) = 135, AQI(CO) = 78, do đó chỉ số AQI của không khí sẽ là giá trị cao nhất là AQI = 135. Giá trị AQI càng cao, không khí càng ô nhiễm.")
    st.write("(+) 0-50 (Tốt) : Không ảnh hưởng.")
    st.write("(+) 51–100 (Trung binh): Một số nhạy cảm có thể bị kích thích nhẹ.")
    st.write("(+) 101–150 (Kém) : 	Nhóm nhạy cảm nên hạn chế ra ngoài.")
    st.write("(+) 151–200 (Xấu) : Nhóm nhạy cảm bị ảnh hưởng rõ, người thường cũng bị ảnh hưởng nhẹ.")
    st.write("(+) 201–300 (Rất xấu) : Ảnh hưởng nghiêm trọng đến sức khỏe.")
    st.write("(+) 301–500 (Nguy hại) : Báo động sức khỏe toàn dân.")
    st.write("-- Công thức tính AQI cho 1 chất :")
    st.latex(r'''
        \text{AQI} = \left( \frac{I_{\text{Hi}} - I_{\text{Lo}}}{C_{\text{Hi}} - C_{\text{Lo}}} \right) \cdot (C - C_{\text{Lo}}) + I_{\text{Lo}}
    ''')
    st.markdown("""
        **Trong đó:**

        - `C`: nồng độ chất đo được
        - `[C_Lo, C_Hi]`: khoảng nồng độ chứa `C`
        - `[I_Lo, I_Hi]`: khoảng AQI tương ứng với `[C_Lo, C_Hi]`
    """)
    st.markdown(
        'Cụ thể cách tính bạn truy cập vào đường dẫn : [Cách tính chỉ số AQI](https://tapchimoitruong.vn/chuyen-muc-3/H%C6%B0%E1%BB%9Bng-d%E1%BA%ABn-m%E1%BB%91i-tr%C6%B0%E1%BB%9Dng-8115)'
    )

    st.write("-- Bên cạnh đó còn có các thông số của thời tiết như nhiệt độ, mức gió, mây mù,... cũng ít nhiều ảnh hưởng đến sự tác động của ô nhiễm lên cơ thể con người.")

    st.divider()