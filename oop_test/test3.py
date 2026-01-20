import pandas as pd
import duckdb

data = {
    'user_id': ['U01', 'U01', 'U01', 'U01', 
                'U02', 'U02', 
                'U03', 'U03', 'U03', 'U04', 'U05'],
    'source': ['Search', 'Search', 'Search', 'Social', 
               'Social', 'Social', 
               'Email', 'Email', 'Email', 'Search', 'Social'],
    'event': ['login', 'click', 'login', 'login', 
              'login', 'click', 
              'login', 'login', 'login', 'login', 'login'],
    'timestamp': pd.to_datetime([
        '2026-01-01 09:00:00', '2026-01-01 09:10:00', '2026-01-03 11:00:00', '2026-01-03 12:00:00',     # U01: 2일 뒤 재방문
        '2026-01-01 10:00:00', '2026-01-01 10:30:00',                                                   # U02: 재방문 없음
        '2026-01-02 12:00:00', '2026-01-02 15:00:00', '2026-01-05 10:00:00', # U03: 당일 재방문 후 3일 뒤 또 방문
        '2026-01-01 08:00:00',                                               # U04: 재방문 없음
        '2026-01-04 14:00:00'                                                # U05: 재방문 없음
    ])
}

# 서비스 기획팀에서 첫 방문(login) 후 다음 방문까지의 간격이 광고 매체별로 어떻게 다른가?
# 1. 순수 재방문 주기 평균 (다른 날 재방문한 경우만)
# 2. 당일 두 번 이상 방문한 유저 수
# 3. 전체 유니크 유저 수
# 4. 재방문을 한 번이라도한 유저 비율
#source  avg_return_days  same_day_repeat_users     total_user_count    retention_rate
#Search  2.0              0                         2                   0.5
#Email   3.0              1                         1                   1.0
#Social  NaN              0                         3                   0.0

df = pd.DataFrame(data)

sql = f"""
        SELECT 
            source,
            AVG(CASE WHEN return_days > 0 THEN return_days END) AS avg_return_days,
            COUNT(DISTINCT CASE WHEN return_days = 0 THEN user_id END) AS same_day_repeat_users,
            COUNT(DISTINCT user_id) AS total_user_count,
            COUNT(DISTINCT CASE WHEN return_days > 0 THEN user_id END) / COUNT(DISTINCT user_id) AS retention_rate
        FROM 
            (
                SELECT 
                    user_id,
                    source,
                    date_diff('day', LAG(timestamp) OVER(PARTITION BY user_id, source ORDER BY timestamp), timestamp) AS return_days
                FROM df
                WHERE event = 'login'
            )    
        GROUP BY source     
    """
sql_report = duckdb.query(sql).to_df()
print(sql_report)

df = df.sort_values(['source','user_id','timestamp']).reset_index(drop=True)
df = df[df['event'] == 'login']

df['logindate'] = pd.to_datetime(df['timestamp'].dt.date)
df['return_days'] = df.groupby(['source','user_id'])['logindate'].diff().dt.days
df['return_days_for_avg'] = df['return_days'].where(df['return_days'] > 0)
df['same_day_users'] = df['user_id'].where(df['return_days'] == 0)
df['retention_users'] = df['user_id'].where(df['return_days'] > 0)

pandas_report = df.groupby('source').agg(
    avg_return_days=('return_days_for_avg', 'mean'),      
    same_day_repeat_users=('same_day_users', 'nunique'),  
    total_user_count=('user_id', 'nunique'),              
    retention_users_count=('retention_users', 'nunique')  
).reset_index()

pandas_report['retention_rate'] = pandas_report['retention_users_count'] / pandas_report['total_user_count']

# 결과 확인
print(pandas_report[['source','avg_return_days','same_day_repeat_users','total_user_count','retention_rate']])

import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정 (환경에 따라 다를 수 있습니다)
plt.rc('font', family='Malgun Gothic') 
plt.rcParams['axes.unicode_minus'] = False

# 시각화를 위한 캔버스 생성 (1행 2열)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 1. 매체별 평균 재방문 주기 (Bar Chart)
sns.barplot(data=sql_report, x='source', y='avg_return_days', ax=axes[0], palette='viridis')
axes[0].set_title('광고 매체별 평균 재방문 주기 (일)', fontsize=15)
axes[0].set_ylabel('평균 재방문일')

# 2. 매체별 리텐션 비율 (Point Chart)
sns.pointplot(data=sql_report, x='source', y='retention_rate', ax=axes[1], color='red')
axes[1].set_title('광고 매체별 리텐션 비율 (Retention Rate)', fontsize=15)
axes[1].set_ylim(0, 1.1) # 비율이므로 0~1 사이로 설정

plt.tight_layout()
plt.show()