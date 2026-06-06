# Whisper/CT2 BPE Token-Level KenLM 1-Pass Fusion 가이드

> 이 문서는 「Fusion Analysis」 문서에서 분리한 **BPE token-level 1-pass LM fusion 연구/구현 브랜치**이다.  
> Word-level 2-pass reranking은 별도 브랜치로 보고, 여기서는 **Whisper decoder의 BPE token 단위에 KenLM prior를 1-pass로 결합하는 방식**만 다룬다.

---

## 0. 목적

Whisper / faster-whisper / CTranslate2(CT2) 기반 ASR에서 **도메인 텍스트만으로 만든 BPE token-level n-gram LM**을 디코딩 중 결합한다.

목표는 기존 reverse trie 기반 phrase bias의 한계를 보완하는 것이다.

reverse trie phrase bias는 특정 phrase prefix가 이미 나온 뒤에 다음 token을 올리는 방식이다. 그래서 **첫 token 진입 자체가 실패하는 도메인 용어**에는 약하다.

BPE token-level LM fusion은 디코딩 중 매 step에서 도메인 LM 점수를 함께 반영하므로, 설계에 따라 첫 token 선택에도 영향을 줄 수 있다.

---

## 1. 범위

### 포함

- BPE token-level KenLM
- Whisper tokenizer 기반 token id corpus 생성
- BPE tokenization augmentation
- KenLM 3-gram / 5-gram 학습
- CT2 beam search 중 1-pass shallow fusion
- ASR top-k only 방식과 ASR top-k ∪ LM/domain 후보 union 방식 비교
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

KenLM은 일반적으로 log10 score를 반환한다. Whisper/CT2 score는 natural log 기준이므로 변환이 필요하다.

```text
LM_logprob_ln = LM_logprob_log10 * ln(10)
final_score = ASR_logprob_ln + alpha * LM_logprob_ln
```

단, 실제 beam search에서는 length penalty / score normalization과의 상호작용이 있으므로, 단순 합산만 보고 판단하면 안 된다. 자세한 내용은 §11을 참고한다.

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
- BPE tokenization mismatch 가능성 존재
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

## 5. BPE tokenization augmentation

Whisper는 BPE 기반 모델이다. 같은 surface text가 디코딩 중 항상 동일한 canonical token path만으로 생성된다고 가정하면, 도메인 LM이 실제 beam search 경로를 충분히 커버하지 못할 수 있다.

다만 여기서 중요한 점은 다음이다.

```text
- 모든 Whisper 계열이 항상 BPE dropout으로 학습됐다고 단정하지 않는다.
- 일부 Whisper 계열, 특히 Large V2 추가 학습에서는 BPE Dropout이 사용된 것으로 알려져 있다.
- Large V3 / turbo / fine-tuned 파생 모델에서는 실제 token path 선호가 모델별로 다를 수 있다.
- 따라서 BPE augmentation은 필수 전제가 아니라 robustness 장치이자 실험 변수로 둔다.
```

즉, 문서의 가정은:

```text
Whisper가 non-canonical path를 항상 적극적으로 낸다
```

가 아니라,

```text
BPE path 다양성 또는 beam search의 비표준 segmentation 가능성에 대비해
LM training corpus를 약간 더 robust하게 만든다
```

이다.

### 예시

```text
"보장개시일"
canonical:
  [A, B, C]

possible alternative:
  [A, X, Y, C]
  [P, Q, R, S]
```

가능한 모든 BPE 조합을 생성하는 것은 비현실적이다. 대신 **stochastic BPE augmentation**을 사용한다.

---

## 6. BPE augmentation 정책

### 기본값 후보

```yaml
bpe_dropout_p: 0.05 ~ 0.15
samples_per_sentence: 2 ~ 4
domain_term_sentence_samples: 4 ~ 8
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

### 반드시 A/B로 검증

BPE augmentation은 효과를 가정하지 말고 아래를 비교한다.

```text
A. canonical only
B. canonical + 2 samples
C. canonical + 4 samples
```

판단 기준:

```text
domain recall 증가
CER 악화 없음
insertion 증가 없음
latency 변화 없음
```

augmentation이 효과가 없으면 canonical only로 되돌린다.

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

### special token 정책

MVP에서는 순수 transcription text token만 LM corpus에 넣는다.

```text
SOT / language / task token:
  LM corpus에서 제외

timestamp token:
  LM scoring 제외

EOT:
  필요하면 별도 token으로 실험
  MVP에서는 제외 가능
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

## 10. KenLM scoring API 주의점

KenLM scoring pseudo-code에서는 문자열을 바로 score하는 것처럼 쓰기 쉽지만, 실제 C++ API에서는 vocabulary index를 거쳐야 한다.

### 초기 state

각 beam 시작 시 KenLM state를 초기화한다.

```cpp
kenlm::State state;
model.BeginSentenceState(&state);
```

필요에 따라 sentence boundary를 무시하고 null-context로 시작하는 설정도 실험할 수 있지만, MVP에서는 `BeginSentenceState`를 기본으로 둔다.

### token id → KenLM word index

BPE token id는 KenLM corpus에서 `t1234` 같은 pseudo-word로 들어갔다. 디코딩 중에도 동일하게 변환한다.

```cpp
std::string word_str = "t" + std::to_string(token_id);
auto word = model.GetVocabulary().Index(word_str);
```

성능상 매번 string을 만들면 느리므로, 실제 구현에서는 시작 시점에 전체 vocab에 대해 mapping table을 만든다.

```cpp
std::vector<WordIndex> token_id_to_kenlm_word;
```

### score 계산

```cpp
kenlm::State next_state;
auto score = model.FullScore(prev_state, word, next_state);

float lm_score_log10 = score.prob;
float lm_score_ln = lm_score_log10 * std::log(10.0f);
```

후보가 선택되면 해당 후보의 `next_state`를 beam state로 보관한다.

---

## 11. length penalty / score normalization

LM score는 token마다 누적된다. 따라서 단순히 아래처럼 더하면:

```text
score = ASR_score + alpha * LM_score
```

hypothesis 길이에 따라 유리하거나 불리해질 수 있다. 이 문제는 insertion / hallucination과 직접 연결된다.

### 권장 score 구조

개념적으로는 다음 항을 분리해서 본다.

```text
final_score =
  ASR_score
  + alpha * LM_score
  + beta * length_penalty
```

다만 CT2에는 기존 beam score normalization / length penalty 로직이 있으므로, 구현 시에는 다음을 명확히 해야 한다.

```text
1. LM score를 token-level cumulative score에 더할지
2. length penalty 적용 전 score에 더할지
3. length penalty 적용 후 score에 더할지
4. ASR score와 LM score를 함께 normalize할지
5. LM score만 별도 평균화할지
```

### 실험에서 반드시 볼 것

```text
length_ratio
insertion rate
repeated text rate
hallucination phrase hit rate
CER/WER
domain recall
```

### 권장 초기 정책

MVP에서는 가장 단순하게:

```text
raw_step_score = ASR_step_score + alpha * LM_step_score
기존 CT2 beam normalization은 그대로 유지
```

로 시작한다.

그 다음 아래를 ablation한다.

```text
A. 기존 CT2 length penalty 그대로
B. LM score도 ASR과 함께 누적 후 기존 normalization
C. LM score를 token count로 평균화
D. 별도 beta length penalty 추가
```

---

## 12. 1-pass fusion 방식

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

### 방식 B: ASR top-k ∪ LM/domain candidate union

ASR 후보와 LM/domain prior가 선호하는 후보를 합쳐서 평가한다.

```text
candidate_set = ASR_topk ∪ LM_or_domain_candidates
```

각 후보에 대해:

```text
final_score = ASR_score + alpha * LM_score
```

장점:

- LM/domain prior가 첫 token 후보를 새로 제안할 수 있음
- reverse trie / ASR top-k only보다 첫 token rescue 가능성이 큼

단점:

- 구현 난이도 증가
- LM top-k candidate를 얻기 위한 별도 자료구조 필요
- GPU logits에서 추가 후보 token의 ASR score를 gather해야 할 수 있음
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

## 13. Union successor index의 현실적 한계

KenLM은 기본적으로 “주어진 candidate word의 score”를 빠르게 계산하는 라이브러리다. 하지만 “현재 history에서 top-k 다음 token을 반환”하는 기능은 직접적으로 강하지 않다.

ARPA에서 단순히 다음 인덱스를 만들 수는 있다.

```text
history -> observed successor tokens
```

예:

```text
(t123, t456) -> [t789, t777, t888]
```

하지만 이 방식은 관측된 n-gram에는 동작해도, 미관측 history에서는 KenLM의 backoff까지 반영한 진짜 top-k successor를 얻기 어렵다.

즉:

```text
완전한 LM top-k successor 생성은 단순 index보다 어렵다.
```

### 현실적 타협

초기 union MVP에서는 완전한 LM top-k 대신 **domain successor candidates**를 추가한다.

예:

```text
ASR top-k
+
도메인 용어/구절의 다음 token 후보
+
자주 관측된 ARPA successor 일부
```

이 경우 기능적으로 reverse trie phrase bias와 가까워지는 부분이 있다. 다만 차이는 다음이다.

```text
reverse trie:
  해당 후보에 고정 bias를 더함

BPE LM union:
  후보를 candidate set에 추가하고,
  최종 점수는 ASR score + alpha * KenLM probability로 계산함
```

즉, union 후보 생성은 trie와 비슷할 수 있지만, 최종 점수는 수동 bias가 아니라 corpus 기반 LM probability를 쓴다.

---

## 14. 추천 개발 순서

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

### Phase 2: ASR top-k ∪ domain successor candidates

목표:

```text
첫 token 진입 실패 개선 가능성 검증
domain term recall 개선 확인
```

설정:

```yaml
beam_size: 5~10
asr_topk: 20~50
domain_successor_topk: 10~30
alpha: [0.05, 0.1, 0.2]
```

핵심:

```text
완전한 LM top-k 대신 domain successor candidate index를 먼저 사용한다.
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

## 15. CT2 구현 포인트

### 15.1 위치

`LogitsProcessor`보다는 beam search scoring 쪽에 가깝다. 이유는 LM fusion은 단순히 한 step의 logits만 바꾸는 것이 아니라, beam별 LM state를 같이 유지해야 하기 때문이다.

필요한 상태:

```text
beam hypothesis
  token sequence
  ASR cumulative score
  LM state
  LM cumulative score
```

### 15.2 per-beam LM state

각 beam은 KenLM state를 하나씩 가진다.

확장 시:

```cpp
WordIndex word = token_id_to_kenlm_word[token_id];

kenlm::State next_state;
auto score = model.FullScore(prev_state, word, next_state);

float lm_score_ln = score.prob * std::log(10.0f);
candidate_score += alpha * lm_score_ln;
candidate.lm_state = next_state;
```

### 15.3 beam reorder

beam search에서는 매 step마다 살아남는 beam이 바뀐다. 이때 LM state도 같은 index로 reorder되어야 한다.

```text
selected_beam_indices
→ token sequence reorder
→ cache reorder
→ LM state reorder
```

### 15.4 GPU/CPU 경계

KenLM은 CPU 라이브러리다. CT2 Whisper decoder logits는 GPU에 있을 수 있다.

따라서 구현에서 가장 위험한 부분은 CPU/GPU sync다.

권장:

```text
1. GPU에서 ASR top-k를 먼저 뽑음
2. top-k token ids와 scores만 CPU로 가져옴
3. CPU에서 KenLM score 계산
4. beam candidate selection 수행
```

단점:

```text
GPU decoding path에 CPU scoring이 개입하므로 p95 latency가 늘 수 있음
```

대안:

```text
- final pass에서만 사용
- beam_size / topk 제한
- candidate 수 제한
- feature flag 제공
```

---

## 16. 성능 비용 추정

### ASR top-k only

비용:

```text
beam_size * asr_topk * KenLM_score_call_per_step
```

예:

```text
beam_size=5
topk=50
→ step당 250 KenLM score
```

C++ KenLM binary 기준으로는 실험해볼 만한 수준이다. 다만 Python callback으로 하면 안 된다. 반드시 CT2 C++ 내부에서 처리해야 한다.

### union 방식

비용:

```text
beam_size * (asr_topk + domain_successor_topk) * score
+
추가 후보의 ASR logit gather 비용
```

주의점:

- GPU logits에서 추가 후보 token score를 가져와야 할 수 있음
- CPU/GPU sync가 늘면 p95 latency가 나빠질 수 있음
- candidate 수를 작게 제한해야 함

권장:

```yaml
asr_topk: 20~50
domain_successor_topk: 10~30
max_candidate_per_beam: 64
```

### full vocab fusion

비용이 커서 production 초기에는 비추천이다.

---

## 17. alpha 설정

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

## 18. 필요한 텍스트량

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

## 19. 학습 데이터 구성

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

## 20. 평가 지표

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
ASR ∪ domain candidate hit rate
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

## 21. 실험 설계

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
domain_successor_topk: 10, 30
alpha: 0.05, 0.1, 0.2
```

목적:

```text
첫 token entry failure 개선 여부 확인
```

### Exp D: length normalization ablation

```yaml
length_policy:
  - ct2_default
  - normalize_asr_lm_together
  - average_lm_by_token_count
  - beta_length_penalty
```

목적:

```text
LM score 누적이 insertion/hallucination에 미치는 영향 확인
```

### Exp E: 비교군

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

## 22. 성공 기준

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

## 23. 리스크

### 23.1 LM이 너무 강함

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
length normalization 조정
```

### 23.2 top-k only 한계

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

### 23.3 BPE path mismatch

증상:

```text
도메인 용어가 LM에 있는데도 디코딩 path와 안 맞음
```

대응:

```text
BPE augmentation A/B
도메인 용어 포함 문장 oversampling
canonical+sample 비교
```

### 23.4 CT2 내부 구현 복잡도

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

### 23.5 union successor의 reverse trie화

증상:

```text
domain successor 후보가 사실상 수동 phrase rule과 비슷해짐
```

대응:

```text
최종 score는 KenLM probability를 사용
manual bias와 비교 ablation
domain successor 후보 개수 제한
LM-only contribution 별도 로깅
```

---

## 24. 추천 최종 로드맵

### Step 1: offline LM 준비

```text
도메인 text 정규화
Whisper tokenizer로 BPE token id 변환
BPE augmentation A/B corpus 생성
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
ASR top-k ∪ domain candidate scoring
첫 token failure 개선 여부 평가
```

### Step 5: production hardening

```text
feature flag
fallback
config validation
latency p95 gate
hallucination gate
length normalization gate
```

---

## 25. 결론

BPE token-level KenLM 1-pass fusion은 reverse trie phrase bias보다 더 일반적인 text prior 방식이다. CT2를 수정할 수 있다면 기술적으로 시도할 가치가 있다.

다만 중요한 점은 다음이다.

```text
ASR top-k only fusion:
  구현은 쉽지만 첫 token rescue는 제한적

ASR top-k ∪ domain 후보 union:
  첫 token 문제에 더 직접적이지만 구현과 latency 부담 증가

full vocab fusion:
  이론상 좋지만 production MVP로는 비추천
```

따라서 추천은:

```text
1. BPE token-level 5-gram KenLM
2. BPE augmentation은 실험 변수로 둠
3. CT2 1-pass ASR top-k only로 skeleton 검증
4. length penalty / score normalization을 반드시 ablation
5. 효과 한계 확인 후 union 후보 확장
6. alpha는 0.05~0.2 중심으로 운영 후보 탐색
```

이 방향은 “text만으로 도메인 prior를 주입한다”는 목표에 맞고, 기존 reverse trie의 첫 token 한계를 보완할 수 있는 현실적인 다음 단계다.
