# 가상화폐 예측 API
## 학습된 가상화폐 모델을 가지고 분마다 새로운 가격 예측을 DB에 저장합니다.
업비트에서 읽어온 최근 데이터 50개 혹은 100개를 가져온다.

미리 학습된 각 LSTM 모델에 수집된 데이터를 LSTM 인풋에 맞춰 넣는다.

예측되어 나온 결과를 db에 저장한다.
**************
## API 실행 과정
![제목 없는 다이어그램 drawio (1)](https://github.com/Digital-coin-predict-Service/BTC-API/assets/112631585/34165e3c-b4f3-4730-9935-dd7e94304073)

빠른 예측을 위해 인공지능 모델을 미리 load 해두고 임의의 값으로 예측을 한 번 한다.

1분마다 업비트에서 최근의 볼륨과 가치(value) 데이터를 가져온다.

인공지능 모델을 학습시킬 때와 같은 방식으로 전처리를 한다.

준비된 모델을 이용하여 예측을 진행한다.

예측 후에는 CoinPredict table에 예측된 coin의 ID, 예측된 시간을 저장한다.

PredictValues table에 CoinPredictId와 예측값을 저장한다.

***************
## 추가 요구 사항

추후에 5분, 30분, 1시간, 일 단위 예측도 추가될 예정이다.

각 예측 시에 여러가지 예측을 할 수 있도록 모델을 수정하여 가격 예측시 여러개의 그래프가 그리는 영역의 히트맵을 줄 것이다.

즉 가격 변동의 확률로 나타날 것이다.
