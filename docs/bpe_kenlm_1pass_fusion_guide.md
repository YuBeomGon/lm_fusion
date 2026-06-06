# Whisper/CT2 BPE Token-Level KenLM 1-Pass Fusion 가이드

## 0. 목적

이 문서는 Whisper / faster-whisper / CTranslate2(CT2) 기반 ASR에서 **도메인 텍스트만으로 만든 BPE token-level n-gram LM**을 디코딩 중에 결합하는 방법을 정리한다.

목표는 기존 reverse trie 기반 phrase bias의 한계를 보완하는 것이다.

reverse trie phrase bias는 특정 phrase prefix가 이미 나온 뒤에 다음 token을 올리는 방식이므로, **첫 token 진입 자체가 실패하는 도메인 용어**에는 약하다. 반면 BPE token-level LM fusion은 디코딩 중 매 step에서 도메인 LM 점수를 함께 반영하므로, 설계에 따라 첫 token 선택에도 영향을 줄 수 있다.

---

## 1. 범위

### 포함

- BPE token-level KenLM
- Whisper tokenizer 기반 token id corpus 생성
- BPE dropout / stochastic tokenization augmentation
- KenLM 3-gram / 5-gram 학습
- CT2 beam search 중 1-pass shallow fusion
- ASR top-k only 방식과 ASR top-k + LM/domain 후보 union 방식 비교
- 실험 설계, 평가 지표, 성능 측정

### 제외

- word-level LM 2-pass reranking
- external neural LM fusion
- text-only Whisper fine-tuning
- LoRA / adapter fine-tuning
- LLM post-correction
- 기존 reverse trie phrase bias 고도화

---

## 2. 기본 아이디어

Whisper decoder는 매 step마다 다음 BPE token 확률을 낸다.

기존 decoding score:

```text
score = ASR_logprob
```

BPE LM fusion 적용 후:

```text
score = ASR_logprob + alpha * LM_logprob
```

여기서:

```text
ASR_logprob = Whisper decoder가 낸 token log probability
LM_logprob  = 도메인 BPE n-gram LM이 준 token log probability
alpha       = LM 영향도
```

KenLM은 보통 log10 score를 반환하므로, Whisper/CT2의 natural log score와 합치려면 변환이 필요하다.

```text
LM_logprob_ln = LM_logprob_log10 * ln(10)
final_score = ASR_logprob_ln + alpha * LM_logprob_ln
```

---

## 3. reverse trie phrase bias와의 차이

### reverse trie phrase bias

```text
현재 suffix가 [A]이면 B에 +bias
현재 suffix가 [A, B]이면 C에 +bias
```

예:

```text
"약침"까지 나온 경우에만 "치료"를 boost
```

장점:

- 구현이 단순함
- latency 영향이 작음
- 특정 용어에 대한 제어성이 좋음

한계:

- 첫 token이 틀리면 발동하지 않음
- 짧은 phrase에 걸면 insertion 위험이 큼
- 등록한 phrase 외 일반화가 약함

### BPE token-level LM fusion

```text
지금까지의 BPE token history를 보고
도메인 LM이 다음 token 확률을 제공
```

예:

```text
"한방 병원 에서" 이후
"첩약", "약침", "치료" 관련 token 확률이 올라갈 수 있음
```

장점:

- 도메인 텍스트 전체에서 prior를 학습
- phrase rule보다 일반화 가능
- 설계에 따라 첫 token 선택에도 영향을 줄 수 있음

한계:

- CT2 beam search 수정 필요
- BPE tokenization 문제 존재
- LM 영향이 강하면 hallucination / insertion 가능

---

## 4. BPE token-level LM을 쓰는 이유

Whisper decoder는 BPE token 단위로 동작한다. 따라서 1-pass fusion을 디코딩 내부에 넣으려면 BPE token-level LM이 가장 직접적으로 맞는다.

word-level LM도 가능하지만, 1-pass에서 쓰려면 word boundary를 추적해야 한다.

```text
BPE + 1-pass:
  decoder token 단위와 직접 호환
  구현은 CT2 beam search 수정 중심

Word + 1-pass:
  word boundary, partial word buffer, delayed scoring 필요
  구현 복잡도 높음
```

이번 MVP는 **BPE token-level LM + 1-pass fusion**으로 간다.

---

## 5. BPE dropout 문제와 대응

Whisper는 BPE 기반 모델이며, 학습 과정에서 BPE dropout 또는 유사한 tokenization variability를 경험했을 수 있다. 따라서 같은 surface text가 항상 하나의 canonical token path로만 디코딩된다고 가정하면 약할 수 있다.

예:

```text
"보장개시일"
-> [A, B, C]
-> [A, X, Y, C]
-> [P, Q, R, S]
```

가능한 모든 BPE 조합을 생성하는 것은 비현실적이다. 대신 **stochastic BPE augmentation**을 사용한다.

### 권장 방식

도메인 텍스트 문장마다 여러 tokenization sample을 만든다.

```text
원문:
후유장해 보험금은 보장개시일 이후 지급됩니다

canonical:
t101 t202 t303 t404

dropout sample 1:
t101 t222 t223 t303 t404

dropout sample 2:
t150 t151 t202 t303 t450
```

KenLM 학습 corpus에는 canonical과 sample들을 모두 넣는다.

---

## 6. BPE augmentation 정책

### 기본값

```yaml
bpe_dropout_p: 0.05 ~ 0.15
samples_per_sentence: 2 ~ 4
domain_term_sentence_samples: 8
max_path_len_ratio: 1.5
```

### 추천

일반 문장:

```text
canonical + 2 samples
```

도메인 용어 포함 문장:

```text
canonical + 4~8 samples
```

중요한 도메인 용어 예:

```text
후유장해
보장개시일
첩약처방
약침치료
치주질환
영구치
특정한방물리요법
경피경근온열요법
```

### 검증

각 sampled token path는 반드시 decode 검증을 통과해야 한다.

```text
decode(token_ids) == original_text
```

실패한 sample은 버린다.

---

## 7. KenLM 학습 corpus 형식

KenLM은 word-level n-gram LM이다. BPE token-level LM을 만들려면 token id를 pseudo-word처럼 표현한다.

예:

```text
t50257 t1234 t8721 t991 t345
t50257 t1234 t444 t555 t991 t345
```

각 `t<ID>`가 KenLM 입장에서는 하나의 단어다.

### 예시 변환

원문:

```text
보장개시일 이후 발생한 경우 보상합니다
```

Whisper tokenizer 결과:

```text
[1234, 5678, 9012, 3456, 7890]
```

KenLM corpus line:

```text
t1234 t5678 t9012 t3456 t7890
```

---

## 8. n-gram order

### 후보

```text
3-gram: 빠르고 안정적인 baseline
5-gram: 메인 후보
7-gram: 초기에는 비추천
```

### 추천

텍스트가 적을 때:

```text
1k~10k 문장:
  3-gram 또는 4-gram
```

텍스트가 어느 정도 있을 때:

```text
10k~100k 문장:
  5-gram 추천
```

BPE augmentation까지 하는 경우:

```text
5-gram 유지
7-gram은 실험용
```

BPE token은 word보다 짧으므로 3-gram은 문맥이 짧을 수 있다. 도메인 phrase 흐름을 잡으려면 5-gram이 더 적절하다.

---

## 9. KenLM 학습 명령

예시:

```bash
lmplz -o 5 < train.bpe_tokens.txt > domain_bpe_5gram.arpa
build_binary domain_bpe_5gram.arpa domain_bpe_5gram.binary
```

KenLM `lmplz`는 n-gram probability와 backoff weight를 포함한 ARPA를 생성한다. 일반적으로 modified Kneser-Ney smoothing이 사용된다.

---

## 10. 1-pass fusion 방식

### 방식 A: ASR top-k only

Whisper가 먼저 top-k 후보를 고른 뒤, 그 후보들에 대해서만 LM score를 더한다.

```text
ASR top-k 후보:
  token a, token b, token c

각 후보에 대해:
  final_score = ASR_score + alpha * LM_score
```

장점:

- 구현 난이도 낮음
- CT2 기존 beam search 흐름에 비교적 쉽게 붙음
- 비용이 작음

단점:

- ASR top-k 안에 정답 token이 없으면 LM이 살릴 수 없음
- 첫 token 진입 실패를 완전히 해결하지 못함
- 효과가 “일반화된 seq bias”에 가까워질 수 있음

### 방식 B: ASR top-k + LM/domain candidate union

ASR 후보와 LM 또는 도메인 successor 후보를 합쳐서 평가한다.

```text
candidate_set = ASR_topk + LM_or_domain_topk
```

각 후보에 대해:

```text
final_score = ASR_score + alpha * LM_score
```

장점:

- LM이 첫 token 후보를 새로 제안할 수 있음
- reverse trie / ASR top-k only보다 첫 token rescue 가능성이 큼

단점:

- 구현 난이도 증가
- LM top-k candidate를 얻기 위한 별도 자료구조 필요
- GPU logits에서 LM 후보 token의 ASR score를 gather해야 할 수 있음
- latency 증가 가능

### 방식 C: full vocab fusion

모든 vocabulary token에 대해 LM score를 계산하고 Whisper score와 합친다.

장점:

- 이론적으로 가장 정확함
- LM prior가 모든 token에 적용됨

단점:

- vocab 전체에 KenLM query를 매 step 수행하면 너무 느림
- CT2 GPU decoding과 맞추기 어렵고 CPU/GPU sync 비용이 커짐
- production MVP로 비추천

---

## 11. 추천 개발 순서

### Phase 1: ASR top-k only fusion

목표:

```text
기능 skeleton 검증
LM state per beam 검증
alpha sweep 검증
latency 영향 측정
```

설정:

```yaml
beam_size: 5
asr_topk: 20~50
ngram_order: 5
alpha: [0.05, 0.1, 0.2, 0.4]
```

주의:

```text
이 단계는 첫-token rescue 효과가 제한적일 수 있음.
효과 검증보다는 구현 안정성 검증에 가깝다.
```

### Phase 2: ASR top-k + LM/domain candidate union

목표:

```text
첫 token 진입 실패 개선 가능성 검증
domain term recall 개선 확인
```

설정:

```yaml
beam_size: 5~10
asr_topk: 20~50
lm_topk: 10~30
alpha: [0.05, 0.1, 0.2, 0.4]
```

핵심:

```text
LM/domain top-k 후보를 얻기 위한 successor index 필요
```

### Phase 3: production optimization

목표:

```text
CPU/GPU sync 최소화
LM state cache 최적화
candidate scoring batching
fallback/feature flag
```

---

## 12. LM top-k 후보 생성 문제

KenLM은 기본적으로 “주어진 candidate word의 score”를 빠르게 계산하는 라이브러리다. 하지만 “현재 history에서 top-k 다음 token을 반환”하는 기능은 직접적으로 강하지 않다.

따라서 union 방식을 하려면 별도 자료구조가 필요하다.

### 선택지

1. ARPA에서 n-gram successor index 생성

```text
history -> top successor tokens
```

예:

```text
(t123, t456) -> [t789, t777, t888]
```

2. 도메인 term trie 기반 successor 후보 추가

```text
현재 suffix가 도메인 term prefix와 맞으면
다음 token 후보를 candidate_set에 추가
```

3. KenLM + custom top successor cache

자주 나오는 LM state/history에 대해 top-k successor를 캐싱한다.

### 추천

초기 union MVP는 다음 조합이 현실적이다.

```text
ASR top-k
+
domain successor candidates
```

즉, 완전한 LM top-k보다는 도메인 용어 중심 successor 후보를 추가한다.

---

## 13. CT2 구현 포인트

### 13.1 위치

`LogitsProcessor`보다는 beam search scoring 쪽에 가깝다. 이유는 LM fusion은 단순히 한 step의 logits만 바꾸는 것이 아니라, beam별 LM state를 같이 유지해야 하기 때문이다.

필요한 상태:

```text
beam hypothesis
  token sequence
  ASR cumulative score
  LM state
  LM cumulative score
```

### 13.2 per-beam LM state

각 beam은 KenLM state를 하나씩 가진다.

확장 시:

```cpp
kenlm::State next_state;
float lm_score_log10 = model.FullScore(prev_state, "t1234", next_state).prob;
float lm_score_ln = lm_score_log10 * log(10.0f);
candidate_score += alpha * lm_score_ln;
candidate.lm_state = next_state;
```

### 13.3 beam reorder

beam search에서는 매 step마다 살아남는 beam이 바뀐다. 이때 LM state도 같은 index로 reorder되어야 한다.

```text
selected_beam_indices
-> token sequence reorder
-> cache reorder
-> LM state reorder
```

### 13.4 special token 처리

Whisper special token은 LM 학습/추론에서 정책을 정해야 한다.

추천:

```text
SOT/language/task token:
  LM corpus에서 제외하거나 고정 prefix로만 사용

timestamp token:
  LM scoring 제외

EOT:
  필요하면 별도 token으로 포함
```

MVP에서는 순수 transcription text token만 LM corpus에 넣고, 디코딩 중 special/timestamp token에는 LM score를 0으로 두는 것이 안전하다.

---

## 14. 성능 비용 추정

### ASR top-k only

비용:

```text
beam_size * asr_topk * KenLM_score_call_per_step
```

예:

```text
beam_size=5
topk=50
-> step당 250 KenLM score
```

C++ KenLM binary 기준으로는 실험해볼 만한 수준이다. 다만 Python callback으로 하면 안 된다. 반드시 CT2 C++ 내부에서 처리해야 한다.

### union 방식

비용:

```text
beam_size * (asr_topk + lm_topk) * score
+
LM 후보의 ASR logit gather 비용
```

주의점:

- GPU logits에서 추가 후보 token score를 가져와야 할 수 있음
- CPU/GPU sync가 늘면 p95 latency가 나빠질 수 있음
- candidate 수를 작게 제한해야 함

권장:

```yaml
asr_topk: 20~50
lm_topk: 10~30
max_candidate_per_beam: 64
```

### full vocab fusion

비용이 커서 production 초기에는 비추천이다.

---

## 15. alpha 설정

LM 영향도는 작게 시작해야 한다.

추천 sweep:

```text
alpha = 0.00
alpha = 0.05
alpha = 0.10
alpha = 0.20
alpha = 0.40
```

강한 alpha는 domain term recall을 올릴 수 있지만 insertion/hallucination을 늘릴 수 있다.

초기 기준:

```text
alpha 0.05~0.1:
  안전한 범위

alpha 0.2:
  효과 확인용

alpha 0.4:
  과발화 위험 체크용
```

---

## 16. 필요한 텍스트량

### smoke test

```text
1k~5k 문장
```

목적:

```text
pipeline이 돌아가는지 확인
```

### 유의미한 domain LM

```text
10k~50k 문장
```

목적:

```text
도메인 용어 주변 패턴 학습
3-gram/5-gram 비교 가능
```

### 안정적인 5-gram

```text
50k~100k+ 문장
```

목적:

```text
5-gram sparse 완화
BPE augmentation 포함 가능
```

텍스트가 적으면 문장을 과도하게 augmentation하기보다, 실제 상담 스크립트/약관/label transcript를 더 모으는 것이 우선이다.

---

## 17. 학습 데이터 구성

추천 source:

```text
1. 실제 label transcript
2. 상담 스크립트
3. 약관/상품 설명 문장
4. 도메인 용어가 포함된 synthetic template 문장
```

단, synthetic template은 과하게 반복하면 LM이 boilerplate를 과발화할 수 있다.

권장 비율:

```text
실제 transcript / 스크립트: 70~90%
synthetic term template: 10~30%
```

도메인 용어가 있는 문장은 oversampling 가능하지만, 같은 문장을 너무 많이 반복하면 insertion 위험이 커진다.

---

## 18. 평가 지표

기본:

```text
CER
WER
latency average / p50 / p95
RTF
```

도메인 용어:

```text
domain term recall
domain term precision
false insertion rate
occurrence-level gain/loss
```

첫 token 문제:

```text
first-token entry recall
n-best oracle domain recall
ASR top-k candidate hit rate
ASR + LM/domain union candidate hit rate
```

안정성:

```text
hallucination phrase hit rate
repeated text rate
length ratio
no-speech hallucination rate
```

운영:

```text
GPU memory
CPU usage
tokens/sec
beam search step latency
```

---

## 19. 실험 설계

### Exp A: baseline

```yaml
model: base / finetune
lm_fusion: off
beam_size: 5
```

### Exp B: BPE LM ASR top-k only

```yaml
ngram: 3, 5
bpe_augmentation: off, canonical+2, canonical+4
alpha: 0.05, 0.1, 0.2, 0.4
asr_topk: 20, 50
```

목적:

```text
LM fusion skeleton 효과 확인
latency 확인
seq bias 대비 차이 확인
```

### Exp C: BPE LM union 후보 확장

```yaml
ngram: 5
asr_topk: 20, 50
lm_topk: 10, 30
alpha: 0.05, 0.1, 0.2
```

목적:

```text
첫 token entry failure 개선 여부 확인
```

### Exp D: 비교군

```yaml
seq_bias: best config
postprocess: on/off
finetune_model: on/off
```

비교해야 할 것:

```text
baseline
seq bias
BPE LM top-k only
BPE LM union
finetune
finetune + BPE LM
```

---

## 20. 성공 기준

BPE LM fusion을 계속할지 판단하는 기준:

```text
domain recall +3~5%p 이상
CER 악화 없음 또는 미미
false insertion 증가 제한
latency p95 증가 허용 범위 이내
no-speech hallucination 증가 없음
```

예:

```text
domain recall: +0.03 이상
CER: +0.002 이하 악화 또는 개선
precision: -0.01 이하 하락
latency p95: +10% 이내
```

---

## 21. 리스크

### 21.1 LM이 너무 강함

증상:

```text
도메인 용어 insertion 증가
짧은 segment hallucination 증가
문장이 LM boilerplate로 끌림
```

대응:

```text
alpha 낮춤
synthetic template 비율 낮춤
short/no-speech segment에서 LM off
final pass에만 적용
```

### 21.2 top-k only 한계

증상:

```text
첫 token entry failure 그대로 유지
```

대응:

```text
ASR top-k 증가
union 후보 확장
beam_size 증가
```

### 21.3 BPE path mismatch

증상:

```text
도메인 용어가 LM에 있는데도 디코딩 path와 안 맞음
```

대응:

```text
BPE dropout augmentation
도메인 용어 포함 문장 oversampling
canonical+sample 비교
```

### 21.4 CT2 내부 구현 복잡도

증상:

```text
beam reorder 시 LM state mismatch
GPU/CPU sync 증가
latency p95 증가
```

대응:

```text
top-k only부터 시작
C++ 내부 구현
feature flag
LM state unit test
latency benchmark
```

---

## 22. 추천 최종 로드맵

### Step 1: offline LM 준비

```text
도메인 text 정규화
Whisper tokenizer로 BPE token id 변환
BPE dropout augmentation
KenLM 3-gram/5-gram 학습
```

### Step 2: CT2 top-k only fusion

```text
beam별 KenLM state 유지
ASR top-k 후보에 LM score 추가
alpha sweep
latency 측정
```

### Step 3: oracle 분석

```text
ASR top-k 안에 정답 token이 있었는지
top-k only가 충분한지 판단
```

### Step 4: union 후보 확장

```text
domain successor candidate index 구축
ASR top-k + LM/domain candidate scoring
첫 token failure 개선 여부 평가
```

### Step 5: production hardening

```text
feature flag
fallback
config validation
latency p95 gate
hallucination gate
```

---

## 23. 결론

BPE token-level KenLM 1-pass fusion은 reverse trie phrase bias보다 더 일반적인 text prior 방식이다. CT2를 수정할 수 있다면 기술적으로 시도할 가치가 있다.

다만 중요한 점은 다음이다.

```text
ASR top-k only fusion:
  구현은 쉽지만 첫 token rescue는 제한적

ASR top-k + LM/domain 후보 union:
  첫 token 문제에 더 직접적이지만 구현과 latency 부담 증가

full vocab fusion:
  이론상 좋지만 production MVP로는 비추천
```

따라서 추천은:

```text
1. BPE token-level 5-gram KenLM
2. BPE dropout augmentation
3. CT2 1-pass ASR top-k only로 skeleton 검증
4. 효과 한계 확인 후 union 후보 확장
5. alpha는 0.05~0.2 중심으로 운영 후보 탐색
```

이 방향은 “text만으로 도메인 prior를 주입한다”는 목표에 맞고, 기존 reverse trie의 첫 token 한계를 보완할 수 있는 현실적인 다음 단계다.
