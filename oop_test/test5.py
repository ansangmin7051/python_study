import pandas as pd
import numpy as np
import duckdb 

# 2025년 하반기 유저 가입 및 활동 로그 데이터
np.random.seed(7)
n_rows = 1000
data = {
    'user_id': [f'U{str(i).zfill(3)}' for i in np.random.randint(1, 101, n_rows)],
    'signup_month': np.random.choice(['2025-07', '2025-08', '2025-09', '2025-10'], n_rows),
    'activity_date': pd.to_datetime('2025-07-01') + pd.to_timedelta(np.random.randint(0, 150, n_rows), unit='D'),
    'purchase_amount': np.random.randint(10000, 100000, n_rows)
}

# 요구사항 (Requirements)
# 가입월별/활동월별 집계:유저의 가입월(signup_month)과 활동월(activity_month)을 기준으로 유니크 사용자 수를 구하세요. (단, 활동월은 가입월보다 크거나 같아야 합니다.)
# 경과 월(Month Passed) 계산: 가입월로부터 몇 달이 지났는지 계산하세요.
# 예: 가입 2025-07, 활동 2025-07 → 0개월 / 활동 2025-08 → 1개월
# 데이터 피벗(Pivot): 가로축은 경과 월, 세로축은 가입월이 되도록 데이터를 재구성하세요.

# 결과자료 (Target DataFrame Shape)
# signup_month	month_passed	user_count	retention_rate
# 2025-07	    0	            50	        1.0 (100%)
# 2025-07	    1	            25	        0.5 (50%)
# ...	        ...	            ...	        ...
# 2025-10	    0	            40	        1.0 (100%)


df_cohort = pd.DataFrame(data)





# 활동월 추출
df_cohort['activity_month'] = df_cohort['activity_date'].dt.strftime('%Y-%m')