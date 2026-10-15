# bigger teachers for cartridges

## 배경

[cartridges](https://arxiv.org/abs/2506.06266)는 긴 문서를 매번 프롬프트에 넣는 대신, 문서마다 작은 kv cache인 cartridge를 학습해 쓰는 방법을 제안. 학습은 self-study로, 모델이 문서를 보고 질문과 답을 만든 뒤 cartridge만 붙인 자기 자신이 문서 일부를 context로 본 자기 자신의 답 분포를 따르게 함.

이때 teacher와 student가 같은 모델인 self distillation임. 그래서 teacher를 더 큰 모델로 바꾸면 cartridge가 나아지는지 봄.

## 요약

- 8b teacher로 학습한 cartridge(cross)는 3b 자신으로 학습한 것(self)보다 나아지지 않음.
- cross는 \<answer> 태그를 더 자주 닫지 못함.
- 태그를 닫지 못한 답은 대부분 최대 길이에 걸리거나 반복함.
- cartridge 성능은 teacher보다 synthesis 양이 좌우함.
- synthesis 65536개에서 icl에 가까워짐.
- 생성 없이 logprob으로 채점해도 self와 cross는 비슷함.
- logprob 채점에서도 cartridge는 icl보다 낮음.

## 한계

- synthesis를 65536개까지만 만들어 cartridges 레포(3×65536개, 약 20만 개)보다 작아, 더 큰 규모에서는 teacher의 효과가 다를 수 있음.
- self의 synthesis 65536개는 양과 cot 비율(0.75→0.2)을 함께 바꿔, 두 효과를 나누지 못함.
- synthesis 16384개는 고치기 전의 context 템플릿("consists of 1 notes.")으로 만들어, 65536개와 context가 조금 다름.
- cartridge는 cartridges 레포를 따라 상관없는 텍스트로 초기화하되, gradient.txt 대신 wikitext-2를 씀.
- 문서 앞부분으로 초기화하면 더 나을 수 있음(cartridges 논문은 longhealth에서 4%p 높다고 보고함, 55.3% vs 51.3%).
- cot_prompt는 cartridges 레포의 것을 그대로 써 \<thinking>, \<response>, \<final> 등 여러 태그로 답하는 데이터를 학습함.
- 이 때문에 \<answer> 태그를 닫기 어려웠을 수 있지만 확인하지는 않음.
- 8b의 답 분포를 3b의 cartridge로 따라가기 어려운 것(capacity gap)인지, teacher 답 자체의 문제인지 나누지 못함.

## 큰 teacher로 학습하면 cartridge가 나아지는지

### 실험 설정

student는 llama 3.2 3b instruct이고, teacher는 student 자신(self)과 llama 3.1 8b instruct(cross). 문서는 cartridges 레포를 따라 longhealth 환자 10명의 기록.

synthesis는 cartridges 레포의 llama 설정을 따름. 환자 정보와 노트 하나를 context로, structuring, summarization, question, use case, creative 다섯 seed prompt 중 하나로 teacher가 temperature 0.6, 최대 512 토큰으로 질문을 만듦. 일부 질문에는 cot_prompt를 붙이고, temperature 0, 최대 1024 토큰으로 답을 만듦. 답 토큰마다 teacher의 top-20 logprob을 저장. 두 teacher 모두 cot 0.75로 16384개, 레포의 llama synthesis와 같은 cot 0.2로 65536개를 만들고, 8b teacher는 cot 0.2로 16384개도 만듦.

cartridge는 2048 토큰이고, 맨 앞 bos는 sink로 고정하고 나머지만 학습. cartridge만 붙인 student가 답 토큰마다 teacher top-20 분포와의 cross entropy를 줄임. adamw, learning rate 2e-2 weight decay 0, batch 64, 2 epoch.

평가는 longhealth 환자 10명의 객관식 200문제에서 base(문서 없음), icl(문서 전체를 프롬프트에 넣음), cartridge를 비교. cartridges 레포를 따라 cot 프롬프트와 temperature 0.3으로 답을 생성하되, 답이 잘리지 않도록 최대 길이는 512 대신 1024 토큰. 채점은 \<answer> 태그 안의 답과 가장 비슷한 선택지로 하고, 태그를 닫지 못하면 오답. seed 3개.

### 결과

![](./figures/Llama-3.2-3B-Instruct/longhealth.png)

self는 cot 0.75, 16384개에서 39.2%, cot 0.2, 65536개에서 47.0%이고, cross는 각각 31.8%, 45.3%로 16384개에서는 self보다 낮고, 65536개에서는 차이(1.7%p)가 seed 간 편차(1~2%p)와 비슷함(base 33.8%, icl 50.0%).

cross의 cot 0.2, 16384개는 35.2%로 65536개보다 확연히 낮아, cross에서 65536개로 오른 것은 cot 비율이 아니라 synthesis 양 때문임.

65536개에서 cross는 답의 19%가 \<answer> 태그를 닫지 못해 self(7%)보다 많고, 그중 75%는 최대 길이에 걸렸고 47%는 같은 줄을 반복함. 태그를 닫은 답만 보면 cross가 56.0%로 self(50.6%)와 icl(51.5%)보다 높지만, 쉬운 문제에 몰렸을 수 있음.

teacher 답이 1024 토큰에서 잘린 탓인지 보려고(8b 9.0%, 3b 6.0%) 잘린 답을 빼고 다시 학습했지만, cross가 태그를 닫은 비율은 80%로 그대로였고 정확도도 비슷함(cross 46.5%, self 47.2%).

> 8b teacher로 바꿔도 cartridge는 나아지지 않고, \<answer> 태그를 닫지 못하는 답만 늘어남. 성능은 teacher보다 synthesis 양이 좌우함.

## 생성 없이 채점해도 같은지

### 실험 설정

cot 지시를 뺀 프롬프트 뒤에 선택지 다섯 개를 각각 붙여 선택지 토큰의 평균 logprob이 가장 높은 것을 답으로 봄. 태그를 닫지 못하는 문제나 샘플링 노이즈 없이 보려는 것.

### 결과

![](./figures/Llama-3.2-3B-Instruct/longhealth_logprob.png)

icl은 55.5%로 base(36.0%)보다 크게 높음. cartridge는 16384개에서 35.0~38.5%로 base와 비슷하고, 65536개에서 self 41.5%, cross 40.5%로 base보다 높지만 icl보다는 낮음.

> 생성 없이 채점해도 self와 cross는 비슷하고, cartridge는 icl에 못 미침.
