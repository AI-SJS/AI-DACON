# 웹 광고 클릭률 예측 AI 경진대회

![Algorithm](https://img.shields.io/badge/Algorithm-Machine%20Learning-blue)
![Category](https://img.shields.io/badge/Category-Timeseries%20Classification-green)
![Dataset](https://img.shields.io/badge/Dataset-Structured%20Data-orange)

## 🌟 월간 데이콘: 웹 광고 클릭률 예측 AI 경진대회

### 설명
이번 월간 데이콘의 주제는 클릭률 예측(Click-Through Rate, CTR) 모델을 개발하는 것입니다.   
CTR 모델은 온라인 광고 산업 및 디지털 마케팅 분야에서 매우 중요한 역할을 하며,   
실제 비즈니스 적용을 통해 매출 증대와 마케팅 효율성을 극대화할 수 있습니다.

CTR 예측을 위한 웹 로그 데이터는 다음과 같은 특징을 갖습니다:
- 대용량 데이터
- 클래스 불균형
- 고차원 데이터 (High Cardinality)

이번 프로젝트의 목표는 웹 로그 데이터를 처리하여 광고 클릭률을 예측하는 AI 모델을 개발하는 것입니다.

---

### 🏆 주제
**광고 클릭률을 예측하는 AI 모델 개발**

### 🧩 데이터
- **7일간의 웹 로그**를 기반으로 **하루 동안의 광고 클릭률**을 예측합니다.

---

## ⚙️ 코드 설명

- `ctr_prediction.py` 파일에 주요 코드가 포함되어 있습니다.
- LightGBM과 VotingClassifier를 활용한 앙상블 모델을 사용합니다.

---

## 📜 코드 파일

### **ctr_prediction.py**

```python
# 1. 라이브러리 import
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier

# 2. 데이터 로딩 및 전처리
n_samples = 10000000  # 사용할 샘플 수
df = pd.read_csv('train.csv', nrows=n_samples)
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

df.info()

# 3. 모델 초기화 및 학습
lgb_model1 = lgb.LGBMClassifier(objective='binary', random_state=42)
lgb_model2 = lgb.LGBMClassifier(objective='binary', random_state=52)
lgb_model3 = lgb.LGBMClassifier(objective='binary', random_state=62)

voting_classifier = VotingClassifier(
    estimators=[('lgb1', lgb_model1), ('lgb2', lgb_model2), ('lgb3', lgb_model3)],
    voting='soft'
)

voting_classifier.fit(df.drop('Click', axis=1), df['Click'])

# 4. 테스트 데이터 로딩 및 전처리
def load_data():
    df = pd.read_csv('test.csv')
    df = df.drop('ID', axis=1)
    df[fill_zero_cols] = df[fill_zero_cols].fillna(0)
    df = df.fillna('NAN')
    float_columns = df.select_dtypes(include=['float64']).columns
    df[float_columns] = df[float_columns].astype('int64')
    object_columns = df.select_dtypes(include=['object']).columns
    df[object_columns] = df[object_columns].astype('category')
    return df

test_df = load_data()

# 5. 예측 및 결과 저장
pred = voting_classifier.predict_proba(test_df)
sample_submission = pd.read_csv('sample_submission.csv')
sample_submission['Click'] = pred[:, 1]
sample_submission.to_csv('1000_lgbm_3.csv', index=False)
