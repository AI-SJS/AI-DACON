# 1. 라이브러리 import
>#### import numpy as np  # 수치 계산을 위한 라이브러리
>
>#### import pandas as pd  # 데이터 조작과 분석을 위한 라이브러리
>
>#### import lightgbm as lgb  # LightGBM 모델을 위한 라이브러리
>
>#### from sklearn.ensemble import VotingClassifier  # 여러 모델을 결합하는 앙상블 기법을 위한 라이브러리

---

# 2. 데이터 로딩 및 전처리
>## 훈련 데이터 CSV 파일에서 읽을 샘플 수를 설정
>#### n_samples = 10000000  # 1천만 개의 샘플을 읽어옴
>
>## 지정된 수의 샘플을 읽어 DataFrame으로 로드
>####  df = pd.read_csv('train.csv', nrows=n_samples)
>
>## 'ID' 열은 학습에 필요하지 않으므로 삭제
>#### df = df.drop('ID', axis=1)
>
>## 특정 열의 결측값을 0으로 채우고, 결측값이 있는 경우 0이 적절한 대체값일 때 사용
>#### df['F04'] = df['F04'].fillna(0)
>#### df['F11'] = df['F11'].fillna(0)
>#### df['F18'] = df['F18'].fillna(0)
>#### df['F19'] = df['F19'].fillna(0)
>#### df['F24'] = df['F24'].fillna(0)
>#### df['F27'] = df['F27'].fillna(0)
>#### df['F29'] = df['F29'].fillna(0)
>#### df['F32'] = df['F32'].fillna(0)
>#### df['F33'] = df['F33'].fillna(0)
>#### df['F36'] = df['F36'].fillna(0)
>#### df['F38'] = df['F38'].fillna(0)
>
>## 남은 모든 결측값을 문자열 'NAN'으로 채우고, 범주형 데이터에서 결측값을 하나의 카테고리로 처리하기 위해 사용
>#### df = df.fillna('NAN')
>
>## float64 타입의 열을 int64 타입으로 변환하며, 메모리 사용량을 줄이고 모델 학습 시 효율성을 높이기 위함
>#### float_columns = df.select_dtypes(include=['float64']).columns
>#### df[float_columns] = df[float_columns].astype('int64')
>
>## object 타입의 열을 'category' 타입으로 변환하며, 메모리 효율성을 높이고 LightGBM 모델이 범주형 데이터를 더 잘 처리할 수 있게 하기 위함
>#### object_columns = df.select_dtypes(include=['object']).columns
>#### df[object_columns] = df[object_columns].astype('category')
>
>## DataFrame의 정보를 출력하여 변환이 제대로 이루어졌는지 확인
>#### df.info()

---

# 3. 모델 초기화 및 학습
>## 세 개의 LightGBM 모델을 서로 다른 random state로 초기화하고, 모델의 다변성을 확보하여 앙상블 성능을 향상시키기 위함
>#### lgb_model1 = lgb.LGBMClassifier(objective='binary', random_state=42)
>#### lgb_model2 = lgb.LGBMClassifier(objective='binary', random_state=52)
>#### lgb_model3 = lgb.LGBMClassifier(objective='binary', random_state=62)
>
>## 세 개의 LightGBM 모델을 결합하는 VotingClassifier를 생성하며, 소프트 보팅(soft voting)을 사용하여 각 모델의 예측 확률을 평균함
>#### voting_classifier = VotingClassifier(
>>####     estimators=[('lgb1', lgb_model1), ('lgb2', lgb_model2), ('lgb3', lgb_model3)], 
>>####     voting='soft'
>#### )
>
>## VotingClassifier를 훈련 데이터에 맞춰 학습시키고, 'Click' 열이 목표 변수라고 가정
>#### voting_classifier.fit(df.drop('Click', axis=1), df['Click'])

---

# 4. 테스트 데이터 로딩 및 전처리
>## 테스트 데이터를 로드하고 전처리하는 함수를 정의
>#### def load_data():
>>    ### 테스트 CSV 파일을 DataFrame으로 읽기
>>####     df = pd.read_csv('test.csv')
>>    
>>    ### 예측에 필요하지 않은 'ID' 열을 삭제
>>    df = df.drop('ID', axis=1)
>>    
>>    ### 특정 열의 결측값을 0으로 채움
>>####     df['F04'] = df['F04'].fillna(0)
>>####     df['F11'] = df['F11'].fillna(0)
>>####     df['F18'] = df['F18'].fillna(0)
>>####     df['F19'] = df['F19'].fillna(0)
>>####     df['F24'] = df['F24'].fillna(0)
>>####     df['F27'] = df['F27'].fillna(0)
>>####     df['F29'] = df['F29'].fillna(0)
>>####     df['F32'] = df['F32'].fillna(0)
>>####     df['F33'] = df['F33'].fillna(0)
>>####     df['F36'] = df['F36'].fillna(0)
>>####     df['F38'] = df['F38'].fillna(0)
>>    
>>    ### 남은 모든 결측값을 문자열 'NAN'으로 채움
>>####     df = df.fillna('NAN')
>>    
>>    ### float64 타입의 열을 int64 타입으로 변환
>>####     float_columns = df.select_dtypes(include=['float64']).columns
>>####     df[float_columns] = df[float_columns].astype('int64')
>>    
>>    ### object 타입의 열을 'category' 타입으로 변환
>>####     object_columns = df.select_dtypes(include=['object']).columns
>>####     df[object_columns] = df[object_columns].astype('category')
>>    
>>####     return df
>>
>## 정의된 함수를 사용하여 테스트 데이터를 로드하고 전처리
>#### test_df = load_data()

---

# 5. 예측 및 결과 저장
>## 학습된 VotingClassifier를 사용하여 테스트 데이터에 대한 확률 예측하고, 각 클래스에 대한 확률을 반환하므로 Click=1에 대한 확률을 사용
>#### pred = voting_classifier.predict_proba(test_df)
>
>## 샘플 제출 CSV 파일을 읽어옴
>#### sample_submission = pd.read_csv('sample_submission.csv')
>
>## 샘플 제출 DataFrame의 'Click' 열에 예측된 확률을 할당하며, 예측 결과의 두 번째 열에는 Click=1에 대한 확률이 포함됨
>#### sample_submission['Click'] = pred[:, 1]
>
>## 업데이트된 샘플 제출 DataFrame을 새로운 CSV 파일로 저장
>#### sample_submission.to_csv('1000_lgbm_3.csv', index=False)
