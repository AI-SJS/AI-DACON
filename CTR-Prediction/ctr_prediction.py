# ================================================================
# 📌 라이브러리 import
# ================================================================
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier

# ================================================================
# 📌 데이터 로딩 및 전처리
# ================================================================
# 사용할 샘플 수 설정
n_samples = 10000000

# 데이터 로드
df = pd.read_csv('train.csv', nrows=n_samples)

# 'ID' 열 제거 (학습에 필요하지 않음)
df = df.drop('ID', axis=1)

# 특정 열의 결측값을 0으로 채움
fill_zero_cols = ['F04', 'F11', 'F18', 'F19', 'F24', 'F27', 'F29', 'F32', 'F33', 'F36', 'F38']
df[fill_zero_cols] = df[fill_zero_cols].fillna(0)

# 나머지 결측값은 'NAN' 문자열로 채움
df = df.fillna('NAN')

# float64 열을 int64로 변환
float_columns = df.select_dtypes(include=['float64']).columns
df[float_columns] = df[float_columns].astype('int64')

# object 열을 category 타입으로 변환
object_columns = df.select_dtypes(include=['object']).columns
df[object_columns] = df[object_columns].astype('category')

# 데이터 정보 출력
df.info()

# ================================================================
# 📌 모델 초기화 및 학습
# ================================================================

# LightGBM 모델 세 개를 서로 다른 랜덤 시드로 초기화
lgb_model1 = lgb.LGBMClassifier(objective='binary', random_state=42)
lgb_model2 = lgb.LGBMClassifier(objective='binary', random_state=52)
lgb_model3 = lgb.LGBMClassifier(objective='binary', random_state=62)

# VotingClassifier를 사용한 앙상블 모델 생성
voting_classifier = VotingClassifier(
    estimators=[
        ('lgb1', lgb_model1), 
        ('lgb2', lgb_model2), 
        ('lgb3', lgb_model3)
    ],
    voting='soft'  # Soft voting을 사용하여 예측 확률을 평균
)

# 모델 학습 (목표 변수: 'Click')
voting_classifier.fit(df.drop('Click', axis=1), df['Click'])

# ================================================================
# 📌 테스트 데이터 로딩 및 전처리 함수 정의
# ================================================================
def load_data():
    """
    테스트 데이터를 로드하고 전처리하는 함수
    """
    # 테스트 데이터 로드
    df = pd.read_csv('test.csv')

    # 'ID' 열 제거
    df = df.drop('ID', axis=1)

    # 특정 열의 결측값을 0으로 채움
    df[fill_zero_cols] = df[fill_zero_cols].fillna(0)

    # 나머지 결측값을 'NAN' 문자열로 채움
    df = df.fillna('NAN')

    # float64 열을 int64로 변환
    float_columns = df.select_dtypes(include=['float64']).columns
    df[float_columns] = df[float_columns].astype('int64')

    # object 열을 category 타입으로 변환
    object_columns = df.select_dtypes(include=['object']).columns
    df[object_columns] = df[object_columns].astype('category')

    return df

# 테스트 데이터 로드 및 전처리
test_df = load_data()

# ================================================================
# 📌 예측 및 결과 저장
# ================================================================
# 테스트 데이터에 대한 예측 확률 계산
pred = voting_classifier.predict_proba(test_df)

# 제출 파일 생성
sample_submission = pd.read_csv('sample_submission.csv')
sample_submission['Click'] = pred[:, 1]  # 'Click=1'에 대한 확률 저장
sample_submission.to_csv('1000_lgbm_3.csv', index=False)

# ================================================================
# 📌 완료 메시지 출력
# ================================================================
print("모델 예측 및 결과 저장이 완료되었습니다: '1000_lgbm_3.csv'")
