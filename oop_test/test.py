import pandas as pd
import numpy as np
import duckdb

# 실무 데이터 (로그 DB에서 갓 뽑아온 형태)
raw_data = pd.DataFrame({
    'log_id': [101, 102, 103, 104, 105, 106, 107],
    'user_id': ['User_A', 'User_A', 'User_A', 'User_B', 'User_B', 'User_C', 'User_A'],
    'content': ['날씨 어때?', '오늘 미세먼지 알려줘', '고마워!', '계좌 잔액 조회', '이체해줘', '배고파', '내일 날씨는?'],
    'timestamp': pd.to_datetime([
        '2026-01-18 10:00', '2026-01-18 10:02', '2026-01-18 10:05',
        '2026-01-18 11:00', '2026-01-18 11:01', '2026-01-18 12:00',
        '2026-01-18 15:00' # User_A가 오후에 다시 들어옴
    ])
})

# "어제 들어온 상담 로그들을 분석해서, 세션당 대화가 3번 이상 오간 우수 세션만 추리고, 각 세션의 첫 질문과 마지막 질문을 붙여서 보고서로 만들어주세요."

raw_data = raw_data.sort_values(['user_id','timestamp']).reset_index(drop=True)
raw_data['gap'] = raw_data.groupby('user_id')['timestamp'].diff().dt.total_seconds().fillna(0)
raw_data['is_new'] = (raw_data['gap']  >= 1800).astype(int)
raw_data['session_id'] = raw_data.groupby('user_id')['is_new'].cumsum()

raw_data['session_count'] = raw_data.groupby(['user_id','session_id'])['user_id'].transform('count')

filter_df = raw_data[raw_data['session_count'] >= 3]

report = filter_df.groupby(['user_id','session_id']).agg(
    first_content = ('content', 'first'),
    last_content = ('content', 'last')
).reset_index()

print(report)

data = {
    'user_id': ['A', 'A', 'A', 
                'A', 'A', 'A', 'A', 'A', 
                'B', 'B', 'B', 
                'C', 'C', 'C', 'C'],
    'event': ['click', 'cart', 'purchase', 
              'click', 'click', 'cart', 'purchase', 'click', 
              'click', 'cart', 'click', 'click', 'cart', 'cart', 'purchase'],
    'timestamp': pd.to_datetime([
        '2026-01-19 10:00:00', '2026-01-19 10:05:00', '2026-01-19 10:10:00', 
        '2026-01-19 15:00:00', '2026-01-19 10:25:00', '2026-01-19 10:30:00', '2026-01-19 10:35:00', '2026-01-19 10:40:00', # A: 한 세션에서 구매 후 오후에 재방문
        '2026-01-19 11:00:00', '2026-01-19 11:10:00', '2026-01-19 11:20:00',                         # B: 구매 없이 이탈
        '2026-01-19 12:00:00', '2026-01-19 12:05:00', '2026-01-19 12:10:00', '2026-01-19 12:15:00'  # C: 한 세션 내에서 구매
    ])
}

# 상황: 마케팅 팀에서 **"장바구니에 담은 후 실제 구매까지 이어진 고객의 행동 패턴"**을 분석해달라는 요청이 들어왔습니다. 
# 단순히 구매한 데이터뿐만 아니라, 구매 전후의 흐름을 파악하는 것이 핵심입니다.

#세션 생성: 사용자별로 이벤트 발생 간격이 1시간(3600초) 이상이면 새로운 세션으로 간주하세요.
#구매 세션 필터링: 여러 세션 중 'purchase'(구매) 이벤트가 포함된 세션만 추출하세요.
#지표 계산: 해당 세션 내에서 'purchase'가 일어나기 전까지 발생한 **총 클릭 횟수(click_count)**와 **장바구니 담기 횟수(cart_count)**를 구하세요.
#최종 보고서: user_id와 session_id별로 첫 이벤트 시간, 구매 시간, 그리고 위에서 계산한 지표들을 포함한 테이블을 만들어주세요.

df = pd.DataFrame(data)

df = df.sort_values(['user_id','timestamp']).reset_index(drop=True)
df['gap'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds().fillna(0)
df['is_new'] = (df['gap']>= 3600).astype(int)
df['session_id'] = df.groupby('user_id')['is_new'].cumsum()

df['is_purchase'] = (df.groupby(['user_id','session_id'])['event'].transform('max') == 'purchase').astype(int)
purchase_df = df[df['is_purchase'] == 1]

purchase_df['click_cnt'] = (purchase_df['event'] == 'click').groupby([purchase_df['user_id'],purchase_df['session_id']]).transform('sum')

purchase_df['cart_cnt'] = (purchase_df['event'] == 'cart').groupby([purchase_df['user_id'],purchase_df['session_id']]).transform('sum')

report1 = purchase_df.groupby(['user_id','session_id']).agg(
    first_event_time = ('timestamp','first'),
    purchase_event_time = ('timestamp','last'),
    click_cnt = ('click_cnt','min'),
    cart_cnt = ('cart_cnt','min')
)

print(report1)

test_df = pd.DataFrame(data)

test_sql = f"""
    SELECT 
        *,
        timestamp - LAG(timestamp) OVER(PARTITION BY user_id ORDER BY timestamp) AS gap
    FROM test_df
    ORDER BY user_id, timestamp
"""
results1 = duckdb.query(test_sql).to_df

print(results1)

sql = f"""
        SELECT 
            user_id,
            session_id,
            purchase_cumsum,
            MIN(timestamp) AS first_event_time,
            MAX(timestamp) AS purchase_event_time,
            SUM(CASE WHEN event = 'click' THEN 1 ELSE 0 END) AS click_cnt,
            SUM(CASE WHEN event = 'click' THEN 1 ELSE 0 END) AS cart_cnt
        FROM 
            (
                SELECT *,
                        SUM(CASE WHEN event = 'purchase' THEN 1 ELSE 0 END)
                        OVER(PARTITION BY user_id, session_id ORDER BY timestamp DESC
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                        ) AS purchase_cumsum
                FROM df
            )   
        WHERE purchase_cumsum > 0
        GROUP BY user_id, session_id, purchase_cumsum   
        ORDER BY user_id, session_id, purchase_cumsum   
      """
results = duckdb.query(sql).to_df

print(results)


