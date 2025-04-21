def calculate_sub_index(C, breakpoints):
    for bp in breakpoints:
        C_lo, C_hi, I_lo, I_hi = bp
        if C_lo <= C <= C_hi:
            return ((I_hi - I_lo) / (C_hi - C_lo)) * (C - C_lo) + I_lo
    return None 


def calculate_aqi(pm25, pm10, o3, no2, so2, co):
    # PM2.5 (µg/m³)
    pm25_bp = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500)
    ]
    
    # PM10 (µg/m³)
    pm10_bp = [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 504, 301, 400),
        (505, 604, 401, 500)
    ]
    
    # O3 (ppb)
    o3_bp = [
        (0, 54, 0, 50),
        (55, 70, 51, 100),
        (71, 85, 101, 150),
        (86, 105, 151, 200),
        (106, 200, 201, 300)
    ]
    
    # NO2 (ppb)
    no2_bp = [
        (0, 53, 0, 50),
        (54, 100, 51, 100),
        (101, 360, 101, 150),
        (361, 649, 151, 200),
        (650, 1249, 201, 300),
        (1250, 1649, 301, 400),
        (1650, 2049, 401, 500)
    ]
    
    # SO2 (ppb)
    so2_bp = [
        (0, 35, 0, 50),
        (36, 75, 51, 100),
        (76, 185, 101, 150),
        (186, 304, 151, 200),
        (305, 604, 201, 300),
        (605, 804, 301, 400),
        (805, 1004, 401, 500)
    ]
    
    # CO (ppm)
    co_bp = [
        (0.0, 4.4, 0, 50),
        (4.5, 9.4, 51, 100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300),
        (30.5, 40.4, 301, 400),
        (40.5, 50.4, 401, 500)
    ]

    sub_indices = {
        'pm25': calculate_sub_index(pm25, pm25_bp),
        'pm10': calculate_sub_index(pm10, pm10_bp),
        'o3': calculate_sub_index(o3, o3_bp),
        'no2': calculate_sub_index(no2, no2_bp),
        'so2': calculate_sub_index(so2, so2_bp),
        'co': calculate_sub_index(co, co_bp)
    }

    aqi = max(sub_indices.values())
    main_pollutant = max(sub_indices, key=sub_indices.get)

    return round(aqi), main_pollutant, sub_indices
