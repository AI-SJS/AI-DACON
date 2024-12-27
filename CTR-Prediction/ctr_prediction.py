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
