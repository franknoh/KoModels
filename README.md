# Pretrained Models for Korean

* 대다수의 모델은 영어만을 다루고 있고, 만약 한국어를 지원하더라도 단독 모델이 아니라 다국어 버전을 통해 지원하는 경우가 많습니다.
* 그렇기에 한국어만을 위한 모델을 학습하였습니다.
* 모델은 주로 jax/flax를 이용하여 구현 및 학습되었으며, 그 후 pytorch로 변환되었습니다.

## Updates

* 2023-02-27: whisper, t5 모델 추가

## Download Link

| Model         | Size  | Link                                                        |
|---------------|-------|-------------------------------------------------------------|
| whisper-tiny  | 39 M  | [link](https://huggingface.co/hyunwoo3235/whisper-tiny.ko)  |
| whisper-small | 244 M | [link](https://huggingface.co/hyunwoo3235/whisper-small.ko) |
| t5 v1.1 small | 77 M  | [link](https://huggingface.co/hyunwoo3235/t5-v1_1-base-ko)  |
| t5 v1.1 base  | 250 M | [link](https://huggingface.co/hyunwoo3235/t5-v1_1-base-ko)  |

## Documentation

각각의 모델에 대한 자세한 설명은 아래 링크를 참고해주세요.

* [whisper](docs/whisper.md)
* [t5](docs/t5.md)

## Acknowledgments

* Research supported with Cloud TPUs from Google's TPU Research Cloud (TRC)
* 일부 모델은 과학기술정보통신부의 재원으로 한국지능정보사회진흥원의 지원을 받아 구축된 데이터셋을 활용하여 개발되었습니다.
  모델 개발에 활용된 데이터는 모델 상세 페이지에서 확인하실 수 있으며 AI 허브(aihub.or.kr)에서 다운로드 받으실 수 있습니다.
