# 📊 웹 광고 클릭률 예측 AI 경진대회 (CTR-Prediction)

![Algorithm](https://img.shields.io/badge/Algorithm-Machine%20Learning-blue)
![Category](https://img.shields.io/badge/Category-Timeseries%20Classification-green)
![Dataset](https://img.shields.io/badge/Dataset-Structured%20Data-orange)

## 🌟 대회 개요
이 프로젝트는 **데이콘의 웹 광고 클릭률 예측 AI 경진대회**를 기반으로 진행되었습니다.  
광고 클릭률(CTR)을 예측하는 AI 모델을 개발하여 디지털 마케팅의 효율성을 극대화하는 것이 목표입니다.

---

### 🏆 주제
**광고 클릭률을 예측하는 AI 모델 개발**

CTR 예측을 위한 웹 로그 데이터는 다음과 같은 특징을 갖습니다:
- **대용량 데이터**: 효율적인 데이터 전처리와 모델 훈련 필요
- **클래스 불균형**: 클릭 데이터의 희소성 해결
- **고차원 데이터**: 고유값이 많은 범주형 데이터 처리

---

### 📂 폴더 구성
- \`ctr_prediction.py\`: 데이터 전처리, 모델 학습 및 예측 코드
---

### ⚙️ 기술 스택
- Python
- LightGBM
- Scikit-learn
- Pandas
- NumPy

---

### ⚙️ 코드 설명
1. **데이터 전처리**
   - 결측값 처리: 특정 열의 결측값을 0으로 채움.
   - 범주형 데이터: \`object\` 타입을 \`category\`로 변환하여 메모리 효율성 증가.
   - 데이터 타입 변환: \`float64\`를 \`int64\`로 변환.

2. **모델 학습**
   - **LightGBM** 기반의 세 가지 모델을 각각 다른 랜덤 시드로 학습.
   - **VotingClassifier**를 사용하여 소프트 보팅(Soft Voting)으로 앙상블.

3. **결과 예측**
   - 테스트 데이터에 대한 확률 예측을 수행.
   - 최종 결과를 \`1000_lgbm_3.csv\`로 저장.

---

### 📊 주요 결과
- **평가지표 (AUC):** 0.762
- **최종 순위:** 82등 / 211 / 상위 38.86%

---

### 📑 데이터 다운로드
데이터는 데이콘에서 제공됩니다. 아래 링크에서 다운로드할 수 있습니다
- [데이콘 웹 광고 클릭률 예측 대회 데이터](https://dacon.io/competitions/official/236258/data)


---

### 📬 문의
더 자세한 내용이 궁금하시다면, GMAIL을 통해 자유롭게 질문해 주세요! 😊
