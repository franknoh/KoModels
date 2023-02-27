# Whisper For Korean

[Whisper](https://arxiv.org/abs/2212.04356)는 encoder-decoder 구조의 범용 음성 인식 모델입니다.

transformers whisper가 jax를 지원하지 않을
당시에 [merge 전 버전](https://github.com/andyehrenberg/transformers/tree/718f53bc839b0efe1741c2ce056945a931643f23)을 사용하여
학습하였습니다.

학습 당시 하나의 발화씩만을 사용하였기에 원 모델에서 지원하는 timestamps는 지원하지 않습니다.

## Hyperparameters

| Model         | Layers | Width | Heads | Batch Size | Learning Rate | Steps | 
|---------------|--------|-------|-------|------------|---------------|-------|
| whisper-tiny  | 4      | 384   | 6     | 256        | 1.5e-3        | 150k  |
| whisper-small | 12     | 768   | 12    | 256        | 1e-3          | 100k  |

## Usage

```python
import librosa

from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperFeatureExtractor
from transformers import AutomaticSpeechRecognitionPipeline

processor = WhisperProcessor.from_pretrained("hyunwoo3235/whisper-tiny.ko")
feature_extractor = WhisperFeatureExtractor.from_pretrained("hyunwoo3235/whisper-tiny.ko")
model = WhisperForConditionalGeneration.from_pretrained("hyunwoo3235/whisper-tiny.ko")

pipe = AutomaticSpeechRecognitionPipeline(
    model=model, feature_extractor=feature_extractor, processor=processor, tokenizer=processor.tokenizer
)

audio, sr = librosa.load("test.wav", sr=16000)
transcription = pipe(audio)
```

## Dataset

해당 모델은 과학기술정보통신부의 재원으로 한국지능정보사회진흥원의 지원을 받아 구축된 데이터를 활용하여 개발되었습니다. 모델 개발에 활용된 데이터는 AI 허브(aihub.or.kr)에서 다운로드 받으실 수 있습니다.

| Dataset             | Size (Hours) | Link                                                                 |
|---------------------|--------------|----------------------------------------------------------------------|
| 한국어 음성              | 1,000        | [link](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=123) |
| 회의 음성               | 3,000        | [link](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=132) |
| 한국인 대화음성            | 4,000        | [link](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=130) |
| 자유대화 음성(일반남여)       | 4,000        | [link](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=109) |
| 한국어 강의 음성           | 4,000        | [link](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=115) |
| 명령어 음성(일반남여)        | 4,000        | [link](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=96)  | 
| 방송 콘텐츠 대화체 음성인식 데이터 | 10,000       | [link](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=463) |
| 숫자가 포함된 패턴 발화 데이터   | 11,842       | [link](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=484) |

## Citation

```
@misc{radford2022whisper,
  doi = {10.48550/ARXIV.2212.04356},
  url = {https://arxiv.org/abs/2212.04356},
  author = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  title = {Robust Speech Recognition via Large-Scale Weak Supervision},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```