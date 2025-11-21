import pandas as pd
import numpy as np
import sys

# --- 1. 설정 ---
# ❗️ (이전에 저장한 파일 이름이 맞는지 확인)
FILE_TO_ANALYZE = 'exps/exp1_redimnet_age_rank/result/age_rank_scores.txt'

# ❗️ 1-A. 나이 그룹을 묶을 단위를 설정합니다 (예: 5년 단위)
AGE_BIN_WIDTH = 5

# ❗️ 1-B. 아웃라이어 기준 배수 (일반적으로 1.5)
IQR_MULTIPLIER = 1.5
# -----------------------------------------------

try:
    # 2. 텍스트 파일을 Pandas DataFrame으로 읽기
    df = pd.read_csv(FILE_TO_ANALYZE, sep='\s+')
    
    # 파일에 'file' 컬럼이 있는지 확인 (출력용)
    has_file_column = 'file' in df.columns

except FileNotFoundError:
    print(f"--- ⚠️ 에러 ---")
    print(f"파일을 찾을 수 없습니다: {FILE_TO_ANALYZE}")
    print("FILE_TO_ANALYZE 변수의 경로를 올바르게 수정했는지 확인하세요.")
    sys.exit()
except pd.errors.EmptyDataError:
    print(f"--- ⚠️ 에러 ---")
    print(f"파일이 비어있습니다: {FILE_TO_ANALYZE}")
    sys.exit()

print(f"--- 📊 1. 기본 통계 (age_rank_scores.txt) ---")
print(f"전체 샘플 수: {len(df)}")

if len(df) == 0:
    print("데이터가 없습니다. 스크립트를 종료합니다.")
    sys.exit()

# 3. "mean값의 수" (평균값 계산)
mean_age = df['age'].mean()
mean_score = df['score'].mean()

print(f"\n[평균값 (Mean)]")
print(f"  - Age Mean:   {mean_age:.4f} 세")
print(f"  - Score Mean: {mean_score:.4f} 점")


# --- 4. 나이 그룹(Bin) 생성 ---
print(f"\n--- 📊 2. 나이 그룹화 (Age Binning) ---")

# 데이터의 최소/최대 나이를 기준으로 bin 경계를 설정
min_age = np.floor(df['age'].min())
max_age = np.ceil(df['age'].max())

# np.arange(20, 81, 5) -> [20, 25, 30, ..., 80]
bins = np.arange(min_age, max_age + AGE_BIN_WIDTH, AGE_BIN_WIDTH)
labels = [f"[{int(bins[i])}-{int(bins[i+1])})" for i in range(len(bins)-1)]

if not labels:
    print(f"[에러] 나이 데이터(min:{min_age}, max:{max_age})로 그룹을 만들 수 없습니다.")
    print("AGE_BIN_WIDTH 설정을 확인하세요.")
    sys.exit()

# 각 샘플이 어떤 나이 그룹에 속하는지 'age_bin' 컬럼에 저장
df['age_bin'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

print(f"  - {AGE_BIN_WIDTH}년 단위로 나이 그룹 생성 (총 {len(labels)}개 그룹)")
print(f"  - 그룹 목록 (일부): {labels[:3]} ... {labels[-3:]}")


# --- 5. "나이 그룹별" 아웃라이어 통계 (새 로직) ---
print(f"\n--- 📊 3. 그룹별 아웃라이어 통계 (New Logic) ---")
print(f"  - 정의: 같은 나이 그룹 내에서 {IQR_MULTIPLIER} * IQR 범위를 벗어난 샘플")

# 5-A. 각 나이 그룹별로 Q1, Q3, IQR을 계산
group_stats = df.groupby('age_bin')['score'].agg(
    Q1=lambda x: x.quantile(0.25),
    Q3=lambda x: x.quantile(0.75)
).reset_index()

# IQR 및 아웃라이어 경계 계산
group_stats['IQR'] = group_stats['Q3'] - group_stats['Q1']
group_stats['lower_bound'] = group_stats['Q1'] - (IQR_MULTIPLIER * group_stats['IQR'])
group_stats['upper_bound'] = group_stats['Q3'] + (IQR_MULTIPLIER * group_stats['IQR'])

# 5-B. 원본 DataFrame에 그룹별 통계(상한/하한)를 다시 병합
df = pd.merge(df, group_stats, on='age_bin', how='left')

# 5-C. 아웃라이어 추출 (샘플의 'score'가 속한 그룹의 'lower_bound'/'upper_bound'를 벗어나는지 확인)
outliers_df = df[
    (df['score'] < df['lower_bound']) | 
    (df['score'] > df['upper_bound'])
].copy() # .copy()를 사용하여 SettingWithCopyWarning 방지

print(f"\n  - 총 아웃라이어 개수: {len(outliers_df)} 개")
if len(df) > 0:
    print(f"  - 아웃라이어 비율: {len(outliers_df) / len(df) * 100:.2f} %")

# 5-D. 아웃라이어 목록 출력 (가장 많이 벗어난 순으로 정렬)
if not outliers_df.empty:
    # 'deviation' 컬럼: 그룹의 경계에서 얼마나 벗어났는지 (양수=상한 초과, 음수=하한 미달)
    def calculate_deviation(row):
        if row['score'] > row['upper_bound']:
            return row['score'] - row['upper_bound']
        elif row['score'] < row['lower_bound']:
            return row['score'] - row['lower_bound']
        return 0

    outliers_df['deviation'] = outliers_df.apply(calculate_deviation, axis=1)
    outliers_df['abs_deviation'] = outliers_df['deviation'].abs()

    print("\n--- 📊 4. 아웃라이어 목록 (가장 많이 벗어난 순 정렬) ---")
    
    # 출력할 컬럼 목록 동적 선택
    display_cols = ['age', 'age_bin', 'score', 'lower_bound', 'upper_bound', 'deviation']
    if has_file_column:
        display_cols.insert(0, 'file') # 'file' 컬럼이 있으면 맨 앞에 추가

    print(outliers_df.sort_values(by='abs_deviation', ascending=False).to_string(
        columns=display_cols,
        float_format="%.4f",
        index=False
    ))
else:
    print("\n  - 발견된 아웃라이어가 없습니다.")