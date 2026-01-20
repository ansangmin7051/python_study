import pandas as pd
import numpy as np
import duckdb

data = {
    'user_id': ['U01', 'U01', 'U01', 'U01', 'U01', 'U01', 
                'U02', 'U02', 'U02', 'U03', 'U03', 'U04', 'U04', 'U04', 'U04', 'U04'],
    'source': ['Google', 'Google', 'Google', 'Google', 'Google', 'Google', 
               'Facebook', 'Facebook', 'Facebook', 'Direct', 'Direct', 'Instagram', 'Instagram', 'Instagram', 'Instagram', 'Instagram'],
    'event': ['login', 'click', 'cart', 'purchase', 'cart', 'purchase',
              'login', 'click', 'click', 'login', 'cart', 'login', 'click', 'cart', 'purchase', 'click'],
    'timestamp': pd.to_datetime([
        '2026-01-20 10:00:00', '2026-01-20 10:05:00', '2026-01-20 10:10:00', '2026-01-20 10:15:00', # U01: 15분 만에 구매
        '2026-01-20 10:20:00', '2026-01-20 10:25:00',
        '2026-01-20 11:00:00', '2026-01-20 11:20:00', '2026-01-20 11:40:00',                        # U02: 구매 없이 이탈
        '2026-01-20 12:00:00', '2026-01-20 12:30:00',                                               # U03: 장바구니만 담고 이탈
        '2026-01-20 13:00:00', '2026-01-20 13:10:00', '2026-01-20 13:20:00', '2026-01-20 13:30:00', '2026-01-20 15:00:00' # U04: 구매 후 재방문
    ])
}

#마케팅 팀에서 광고(Ads)를 통해 들어온 고객이 가입 후 첫 구매까지 평균적으로 몇 분이 걸리는지
#첫 구매 직전에 어떤 행동을 가장 많이 하는지
#user_id	source	    total_clicks	total_carts	duration_min
#U01	    Google	    1	            1	        15.0
#U04	    Instagram	1	            1	        30.0

df = pd.DataFrame(data)

sql = f"""
        SELECT 
            user_id,
            source,
            SUM(CASE WHEN event = 'click' THEN 1 ELSE 0 END) AS total_clicks,
            SUM(CASE WHEN event = 'cart' THEN 1 ELSE 0 END) AS total_carts,
            date_diff('minute', MIN(timestamp), MAX(timestamp)) AS duration_min
        FROM
            (
                SELECT 
                    *
                    ,SUM(CASE WHEN event = 'purchase' THEN 1 ELSE 0 END) OVER(PARTITION BY user_id) AS purchase_total_cnt
                    ,SUM(CASE WHEN event = 'purchase' THEN 1 ELSE 0 END)
                    OVER(
                        PARTITION BY user_id 
                        ORDER BY timestamp ASC
                        ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
                    ) AS purchase_cumsum
                FROM df
                WHERE source != 'Direct'
            )
        WHERE purchase_cumsum > 0  
        AND purchase_total_cnt = purchase_cumsum   
        GROUP BY user_id, source     
        ORDER BY user_id
    """

sql_report = duckdb.query(sql).to_df()
print(sql_report)

df = df[df['source'] != 'Direct']
df = df.sort_values(['user_id','timestamp']).reset_index(drop= True)

df['is_purchase'] = (df['event'] == 'purchase').groupby(df['user_id']).cumsum()
df['first_purchase'] = df.groupby('user_id')['is_purchase'].shift(fill_value=0) == 0
fdf = df[(df['first_purchase']) & (df.groupby('user_id')['is_purchase'].transform('max') > 0) ]

fdf['click_cnt'] = fdf['event'].eq('click').astype(int)
fdf['cart_cnt'] = fdf['event'].eq('cart').astype(int)

pandas_report = fdf.groupby(['user_id','source']).agg(
    total_clicks = ('click_cnt','sum'),
    total_carts = ('cart_cnt', 'sum'),
    max_timestamp = ('timestamp', 'max'),
    min_timestamp = ('timestamp', 'min')
).reset_index()

pandas_report['duration_min'] = (pandas_report['max_timestamp'] - pandas_report['min_timestamp']).dt.total_seconds() / 60

print(pandas_report[['user_id','source','total_clicks','total_carts','duration_min']])
