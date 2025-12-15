# 음성 데이터에서 폭력 감지를 위한 1D-2D 비교 및 모델 최적화


---

현대 사회에서 일상 속 폭력 상황이 빈번하게 발생하면서, 이를 실시간으로 탐지하는 기술의 필요성이 커지고 있다. 기존 영상 기반 방법은 CCTV 사각지대 등에서 한계가 존재하지만, 오디오는 이러한 시각적 제약을 극복할 수 있다.
본 연구는 음성 데이터를 활용하여 일상생활 속 폭력적 소리와 비폭력적 소리를 실시간으로 구분하는 모델을 제안하며,
1D·2D 모델 비교와 최적화를 통해 다양한 환경에서 높은 성능과 안정성을 달성하고자 한다.

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) ![Librosa](https://img.shields.io/badge/Librosa-000000?style=for-the-badge&logo=python&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white) ![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

---

## 📰 데이터 구성


### 1️. 데이터 수집
- **영어 데이터**: *Kaggle – Audio-based Violence Detection Dataset* (일부만 사용)    [데이터 링크](http://kaggle.com/datasets/fangfangz/audio-based-violence-detection-dataset)
- **한국어 데이터**: *YouTube 직접 수집* (실생활 대화·뉴스 음성 활용, 폭력 음성 추출)

### 2. 데이터 전처리
1. **파일 형식 변환:** `mp3` → `wav`  
2. **노이즈/무음 제거:** noisereduce + pydub 사용  
3. **길이 통일:** 3초 단위  
4. **데이터 분할:** Train / Validation / Test

#### 폭력 데이터
| 구분 | Train | Validation | Test |
|------|-------|------------|------|
| 개수 | 1,014 | 370        | 67   |

#### 비폭력 데이터
| 구분 | Train | Validation | Test |
|------|-------|------------|------|
| 개수 | 832   | 244        | 76   |

---

## 📰 2D 데이터 전처리: 스펙트로그램 변환

1D 오디오 데이터를 2D 이미지 형태로 변환하여 2D CNN 모델의 학습 성능을 극대화하는 과정에서 **멜 스펙트로그램(Mel Spectrogram)**을 사용했다.

### Mel Spectrogram 사용 이유
- 인간 청각은 저주파 대역에 더 민감하게 반응하며, Mel Scale은 이러한 청각 특성을 모방하여 저주파 영역에서 더 높은 해상도를 제공한다.
- 일상에서 발생하는 폭력 소리에서 중요한 저주파 특징을 효과적으로 분석할 수 있다.

따라서 본 연구에서는 폭력 오디오 데이터의 특징 추출을 위해 멜 스펙트로그램을 활용했다.

### 스펙트로그램 이미지
<img width="212" height="125" alt="image" src="https://github.com/user-attachments/assets/df93ddab-b9c1-4fd9-bea1-0048a1f4ec0d" />


---

## 📰 1D vs 2D 모델 학습 및 비교

본연구의 오디오 기반 데이터와 멜 스펙트로그램을 활용한 데이터가 ResNet18 사전학습 가중치와는 특성이 맞지 않는다고 판단하여 ResNet18 모델을 기반으로 모든 가중치를 처음부터 학습시켜 오디오 데이터 모델을 구축했다.

1D 오디오 데이터와 2D 스펙트로그램 기반 데이터를 사용한 Residual Learning 기반 CNN 모델의 성능을 비교하기 위해, 두 모델의 파라미터 수를 약 400만 개로 동일하게 설정하고 학습을 진행했다. 이렇게 동일한 조건을 적용해 데이터 형태에 따른 모델 성능 차이를 공정하게 평가했다.

### 1D 모델 구조 [코드](https://github.com/kimjiwoo0707/violence-prediction-ai-model/blob/62bf8dea9c4ce29a02b116afd210fa2bd156ecfe/1D/model_1d.py)

- 입력: 1채널 오디오 신호  
- Conv1d → 64 출력 채널  
- BatchNorm1d + MaxPool1d → 정규화 및 다운샘플링  
- Residual Block: 2개의 Conv1d + BatchNorm1d, 총 37개 레이어  
- 입력과 출력 합산 방식으로 잔차 학습 진행

### 2D 모델 구조 [코드](./2D_model.py)
- 입력: 1D 오디오 데이터를 STFT(Short Time Fourier Transform)로 2D 이미지 형태 변환  
- Conv2d 레이어로 1D 모델과 동일하게 잔차 블록 구성  
- 파라미터 수 맞추기 위해 입력 이미지를 38개 출력 채널로 변환  

### 학습 및 성능 비교
- 하이퍼파라미터 동일하게 설정  
- 학습률(Learning Rate) 변경 후 성능 평가  
- 결과: 학습률 0.001, 0.005에서 2D 모델 성능이 85.31로 1D보다 높음

### 성능 비교
| Model | Learning Rate | Accuracy (%) |
|-------|---------------|-------------|
| 1D    | 0.001         | 79.02       |
| 2D    | 0.001         | **83.92**   |
| 1D    | 0.005         | 59.44       |
| 2D    | 0.005         | **85.31**   |

---

## 📰 최적화 실험 및 성능 향상

2D 모델의 하이퍼파라미터를 추가적으로 조정하고, 데이터 증강 기법을 적용하여 최적의 성능을 찾기 위한 실험을 진행했다.

- **학습 설정**  
  - 학습률(Learning Rate): 0.003  
  - 학습 스케줄러: StepLR  
  - 옵티마이저: Adam (적응형 학습률로 빠른 수렴과 안정적인 학습 가능)

- **데이터 증강 기법**  
  - TimeMasking 적용  
  - `time_mask_param = 40`일 때 가장 우수한 성능 달성

- **채널 확장**  
  - 입력 이미지를 이전 38채널에서 64채널로 변경  
  - 네트워크가 더 복잡한 특징을 학습하도록 지원

- **최종 결과**  
  - 지표: Accuracy, Loss, Confusion Matrix  
  - 최종 Test Accuracy: **90.91%**

Confusion Matrix  
<img width="736" height="248" alt="image" src="https://github.com/user-attachments/assets/053df9ca-da1c-494e-9930-6d49df0a9aaa" />


---

## 📰 최종 정리

본 연구는 오디오 기반 폭력 탐지 모델을 1D·2D 형태로 비교하고 최적화하여, 멜 스펙트로그램을 활용한 2D CNN 모델이 더 높은 정확도와 안정성을 보인다는 점을 확인하였다. 데이터 전처리, Residual Learning 구조 설계, 하이퍼파라미터 조정을 거쳐 최종적으로 90.91%의 정확도를 달성하며 오디오 기반 폭력 탐지의 가능성을 보였다.

---

## 📰 한계
- 데이터 다양성 부족 : 직접 수집한 한국어 데이터의 환경과 발화자 다양성이 부족해 일반화 성능에 제한이 존재한다.

- 실시간 적용성 검증 부족 : 학습 및 테스트 중심으로 진행되어 실시간 환경에서의 처리 속도와 안정성 검증이 필요하다.

- 모델 확장 가능성 : ResNet 기반 CNN 이외에도 Transformer 기반 오디오 모델 등이 성능 향상을 기대할 수 있다.
