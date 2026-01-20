import pandas as pd
import numpy as np
import duckdb
# 실무 데이터 (로그 DB에서 갓 뽑아온 형태)
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

# "어제 들어온 상담 로그들을 분석해서, 세션당 대화가 3번 이상 오간 우수 세션만 추리고, 각 세션의 첫 질문과 마지막 질문을 붙여서 보고서로 만들어주세요."

df = pd.DataFrame(data)

df = df.sort_values(['user_id','timestamp']).reset_index(drop=True)
df['gap'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds().fillna(0)
df['is_new'] = (df['gap'] >= 3600).astype(int)
df['session_id'] = df.groupby('user_id')['is_new'].cumsum()

# 3. 구매가 포함된 세션 식별 (Boolean Masking 활용)
# 세션 내에 'purchase'가 한 번이라도 있는지 확인

df['is_purchase'] = (df['event'] == 'purchase').groupby([df['user_id'], df['session_id']]).transform('any')

fdt = df[df['is_purchase']].copy()

# 5. 구매 시점 이전 데이터만 남기기 (중요: 구매 전 행동 분석을 위함)
# 세션별로 역순으로 누적 합계를 구해 'purchase' 이후 데이터 제외 (SQL의 역순 ROWS BETWEEN)

fdt['purchase_cumsum'] = (
    fdt['event'].eq('purchase')
    .iloc[::-1]
    .groupby([fdt['user_id'], fdt['session_id']])
    .cumsum()
    .iloc[::-1]
)

ffdt = fdt[fdt['purchase_cumsum'] > 0]

ffdt['click_cnt'] = ffdt['event'].eq('click')
ffdt['cart_cnt'] = ffdt['event'].eq('cart')

report = ffdt.groupby(['user_id','purchase_cumsum']).agg(
    first_time = ('timestamp','first'),
    purchase_time = ('timestamp','last'),
    click_count = ('click_cnt','sum'),
    cart_count = ('cart_cnt','sum')
).reset_index()

print(report)
