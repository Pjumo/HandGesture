스켈레톤과 도플러효과를 활용한 제스처 인식
===============
# 작품개요
* 기존의 카메라 기반 제스처인식 모델의 단점
  * 너무 밝거나 어두운 환경에서 제스처 인식 어려움
  * 제스처의 속도가 너무 빠르면 제스처 인식이 어려움
  * 손이 장애물에 가려져 있을 때 제스처 인식이 어려움

* Radar를 통해 열악한 환경에서도 제스처 인식 가능
  * Radar에서 얻은 Doppler 정보로 각각의 제스처 구분 가능
<img width="80%" src="https://github.com/user-attachments/assets/3af1aacf-060a-408b-9df7-edd2932407e5"/>

* 스켈레톤 도출을 통한 제스처인식
  * MediaPipe의 손 좌표와 Visibility 그리고 각 노드간 Angle 정보를 이용하여 베이스 머신러닝 모델 제작
  * Radar로 수집한 Doppler 정보를 이용한 서브 모델 제작 및 앙상블

# 작품내용
* 제스처 라벨링
  * 0 : Nothing (지정된 제스처 이외의 모든 동작)
  * 1 : Click (검지와 엄지 맞닿게 한번)
  * 2 : DoubleClick (클릭 동작 두번 빠르게 반복)
  * 3 : Capture (손을 편 채로 주먹 쥐었다 펴기)
  * 4 : AltTab (손을 편 채로 수평으로 회전 한번)
  * 5 : AltF4 (손을 편 채로 앞으로 내리기)

* 데이터 수집 방법
<img width="80%" src="https://github.com/user-attachments/assets/e8437feb-9306-4b83-b5bc-9e9cf4d8fb69"/>

# 시연영상
시연1 파일 참고
