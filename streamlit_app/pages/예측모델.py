import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap
import matplotlib.pyplot as plt

st.set_page_config(page_title="예측모델", layout="wide")

st.title("예측모델")

# 초기 지도 생성 (한국 중심, 시 단위로 확대)
m = folium.Map(location=[36.5, 127.5], zoom_start=10)

# 클릭한 위치를 저장할 세션 상태 초기화
if 'clicked_points' not in st.session_state:
    st.session_state.clicked_points = []

# 현재 위치 표시
st.subheader("현재 위치")
current_col1, current_col2 = st.columns(2)
with current_col1:
    current_lat = st.number_input("현재 위도", value=36.5, min_value=33.0, max_value=38.0, step=0.01)
with current_col2:
    current_lng = st.number_input("현재 경도", value=127.5, min_value=126.0, max_value=129.0, step=0.01)

# 현재 위치를 중심으로 지도 이동
m.location = [current_lat, current_lng]

# 구분선 추가
st.markdown("---")

# 새로운 발전소 추가
st.subheader("새로운 발전소 추가")

# 위도/경도 입력
col1, col2 = st.columns(2)
with col1:
    lat = st.number_input("위도", value=36.5, min_value=33.0, max_value=38.0, step=0.01)
with col2:
    lng = st.number_input("경도", value=127.5, min_value=126.0, max_value=129.0, step=0.01)

# 풍향/풍속 입력
col3, col4 = st.columns(2)
with col3:
    wind_direction = st.slider("풍향 (도)", min_value=0, max_value=360, value=0, step=45)
with col4:
    wind_speed = st.slider("풍속 (m/s)", min_value=0, max_value=20, value=5, step=1)

# 발전소 추가 버튼
if st.button("발전소 추가"):
    st.session_state.clicked_points.append((lat, lng, wind_direction, wind_speed))

# 영향도 계산 함수 (풍향과 풍속 고려)
def calculate_impact_radius(lat, lng, wind_direction, wind_speed):
    # 기본 영향 반경 (2-5km)
    base_radius = np.random.uniform(2, 5)
    
    # 풍속에 따른 영향 반경 조정
    wind_factor = 1 + (wind_speed / 10)  # 풍속이 빠를수록 영향 반경 증가
    radius = base_radius * wind_factor
    
    # 영향도 감소율 (거리에 따라 감소)
    impact_points = []
    
    # 더 많은 포인트로 부드러운 그라데이션 생성
    for r in np.linspace(0, radius, 100):
        for theta in np.linspace(0, 2*np.pi, 100):
            # 풍향을 고려한 각도 계산 (라디안으로 변환)
            wind_rad = np.radians(wind_direction)
            adjusted_theta = theta + wind_rad
            
            # 타원형 분포를 위한 반경 조정
            # 풍향 방향으로 더 멀리 퍼지도록 조정
            r_adjusted = r * (1 + 0.5 * np.cos(theta - wind_rad))
            
            impact_lat = lat + (r_adjusted * np.cos(adjusted_theta)) / 111.32
            impact_lng = lng + (r_adjusted * np.sin(adjusted_theta)) / (111.32 * np.cos(np.radians(lat)))
            
            # 거리에 따른 영향도 감소 (더 부드러운 감소)
            impact = np.exp(-r / (radius * 0.5))
            
            # 풍향 반대 방향으로 영향도 감소
            wind_effect = 1 - 0.5 * np.cos(theta - wind_rad)
            impact *= wind_effect
            
            impact_points.append([impact_lat, impact_lng, impact])
    
    return impact_points, radius

# 지도에 영향도 표시
for point in st.session_state.clicked_points:
    lat, lng, wind_dir, wind_spd = point
    # 마커 추가
    folium.Marker(
        location=[lat, lng],
        popup=f"새로운 발전소 위치\n풍향: {wind_dir}°\n풍속: {wind_spd}m/s",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    # 영향도 히트맵 추가
    impact_points, radius = calculate_impact_radius(lat, lng, wind_dir, wind_spd)
    HeatMap(impact_points, radius=8, blur=15, max_zoom=1).add_to(m)
    
    # 영향 반경을 타원으로 표시
    folium.Circle(
        location=[lat, lng],
        radius=radius * 1000,  # km를 m로 변환
        color='red',
        fill=False,
        weight=2,
        popup=f'영향 반경: {radius:.1f}km\n풍향: {wind_dir}°\n풍속: {wind_spd}m/s'
    ).add_to(m)

# 지도 표시
folium_static(m, width=1200, height=800)

# 설명 추가
st.markdown("""
### 사용 방법
1. 현재 위치의 위도와 경도를 입력하세요.
2. 새로운 발전소를 추가하려면 위도와 경도를 입력하세요.
3. 풍향과 풍속을 조절하세요.
   - 풍향: 0°는 북쪽, 90°는 동쪽, 180°는 남쪽, 270°는 서쪽
   - 풍속: m/s 단위로 입력
4. "발전소 추가" 버튼을 클릭하면 해당 위치에 새로운 발전소가 추가됩니다.
5. 발전소 주변의 영향도가 풍향과 풍속을 고려하여 그라데이션으로 표시됩니다.
6. 빨간색 마커는 발전소 위치를 나타냅니다.
7. 영향도는 거리에 따라 감소하며, 빨간색이 가장 강한 영향을 나타냅니다.
""")

# 디버깅을 위한 클릭된 위치 표시
if st.session_state.clicked_points:
    st.write("추가된 발전소 위치:", st.session_state.clicked_points)

st.markdown('''
- 아래 지도는 예측모델이 산출한 오염물질 분포(임의값)를 컬러맵으로 시각화한 예시입니다.
- 실제 예측값이 있다면 해당 배열로 Z값을 대체하면 됩니다.
''')

# 위도/경도 격자 생성 (대한민국 영역)
x = np.linspace(126, 130, 100)  # 경도
y = np.linspace(34, 39, 100)    # 위도
X, Y = np.meshgrid(x, y)

# 임의의 예측값(2D Gaussian + 노이즈)
Z = np.exp(-((X-128)**2 + (Y-36.5)**2)/0.5) + 0.05*np.random.rand(*X.shape)

fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(X, Y, Z, levels=20, cmap='jet')
cbar = plt.colorbar(contour, ax=ax, label='예측값(μg/m³)')
ax.set_title('예측모델 기반 오염도 분포')
ax.set_xlabel('경도')
ax.set_ylabel('위도')

# 주요 도시 표시 (예시)
cities = {'Seoul': (37.5665, 126.9780), 'Busan': (35.1796, 129.0756), 'Daegu': (35.8714, 128.6014)}
for name, (lat, lng) in cities.items():
    ax.plot(lng, lat, 'wo')
    ax.text(lng, lat, name, color='white', fontsize=9, ha='center', va='bottom')

st.pyplot(fig)

st.markdown('''
#### 사용법
- 실제 예측값이 있다면 Z 배열을 해당 값으로 대체하면 됩니다.
- 지도 위에 행정구역 경계선, 발전소 위치 등도 추가로 표시할 수 있습니다.
''') 