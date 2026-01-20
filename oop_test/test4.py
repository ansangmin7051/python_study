import pandas as pd
import numpy as np
import duckdb
import matplotlib.pyplot as plt
import seaborn as sns

# 2026년 1월 20일 하루 동안의 서비스 로그 데이터 시뮬레이션
np.random.seed(42)
data = {
    'log_id': range(1, 101),
    'user_id': [f'U{str(i).zfill(3)}' for i in np.random.randint(1, 21, 100)],
    'timestamp': pd.to_datetime('2026-01-20') + pd.to_timedelta(np.random.randint(0, 24, 100), unit='h') + pd.to_timedelta(np.random.randint(0, 60, 100), unit='m'),
    'event': np.random.choice(['view', 'click', 'purchase'], 100, p=[0.5, 0.3, 0.2]),
    'amount': np.random.choice([0, 10000, 20000, 50000], 100, p=[0.8, 0.1, 0.07, 0.03])
}

#[요구사항]
#시간대별 집계: 0시부터 23시까지 각 **시간대(Hour)**별로 유니크 사용자 수(DAU), 총 이벤트 횟수, 총 매출액을 구하세요.
#구매 전환율 계산: 각 시간대별로 전체 이벤트 대비 purchase 이벤트의 비중을 계산하여 conversion_rate 컬럼을 만드세요.
#시각화 준비: 최종 결과물은 시간대순(0~23)으로 정렬된 데이터프레임이어야 합니다.

#[결과자료]
#hour	unique_users	event_count	    total_sales	            conversion_rate
#0	    5	            8	            20000	                0.125
#1	    3	            4	            0	                    0.000
#...	...	            ...	            ...                     ...	
#23	    4	            6	            50000	                0.166

df = pd.DataFrame(data)
sql = f"""
    SELECT 
        HOUR(timestamp) AS timestamp_hour,
        COUNT(DISTINCT user_id) AS unique_users,
        COUNT(event) AS event_count,
        SUM(amount) AS total_sales,
        CAST(SUM(CASE WHEN event = 'purchase' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100 AS conversion_rate
    FROM df
    GROUP BY timestamp_hour
    ORDER BY timestamp_hour
"""

sql_report = duckdb.query(sql).to_df()
print(sql_report)

# 한글 폰트 설정
plt.rc('font', family='Malgun Gothic') 
plt.rcParams['axes.unicode_minus'] = False

# 1. 캔버스 생성
fig, ax1 = plt.subplots(figsize=(12, 6))

# 2. 막대 그래프 (Total Sales) - 왼쪽 Y축
sns.barplot(data=sql_report, x='timestamp_hour', y='total_sales', alpha=0.6, color='skyblue', ax=ax1)
ax1.set_ylabel('총 매출액 (Total Sales)', fontsize=12, color='blue')
ax1.set_xlabel('시간대 (Hour)', fontsize=12)
ax1.set_title('시간대별 매출 현황 및 구매 전환율 추이', fontsize=16)

# 3. 선 그래프 (Conversion Rate) - 오른쪽 Y축 (이중 축 사용)
ax2 = ax1.twinx() # X축을 공유하는 두 번째 Y축 생성
sns.lineplot(data=sql_report, x=range(len(sql_report)), y='conversion_rate', color='red', marker='o', ax=ax2, linewidth=2)
ax2.set_ylabel('구매 전환율 (CVR, %)', fontsize=12, color='red')
ax2.set_ylim(0, 100) # 퍼센트이므로 0~100 범위 권장

plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.show()