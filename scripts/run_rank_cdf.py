# scripts/run_rank_cdf.py
"""정답 도메인 용어의 '첫 subword'가 ASR logits에서 몇 위였는지 분포(CDF).

teacher forcing: decoder_input_ids = reference, 각 position logits에서
다음 정답 토큰의 rank를 구한다. 용어 첫 subword position만 모은다.
"""
import argparse, json, os, yaml, torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from bpe_lm_fusion.data import load_dataset, audio_to_array
from bpe_lm_fusion.domain_terms import load_terms
from bpe_lm_fusion.normalize import normalize_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/poc.yaml")
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    proc = WhisperProcessor.from_pretrained(cfg["model_name"])
    model = WhisperForConditionalGeneration.from_pretrained(
        cfg["model_name"], torch_dtype=torch.float16).to("cuda").eval()  # fp16: GPU VRAM
    tok = proc.tokenizer
    tok.set_prefix_tokens(language=cfg["language"], task=cfg["task"], predict_timestamps=False)

    ds = load_dataset(cfg["dataset_path"])
    test = ds[cfg["test_split"]].select(range(min(args.limit, len(ds[cfg["test_split"]]))))
    terms = load_terms(cfg["domain_terms_file"])
    # 용어를 in-context 토큰열로 준비. Whisper byte-level BPE는 단어 시작에 공백마커가 붙어,
    # 단독 토큰화와 문장 중간(앞 공백) 토큰화의 첫 subword id가 다르다.
    # -> 앞공백 포함/미포함 두 변형을 모두 후보로 두고, label 토큰열에서 '부분열'로 매칭한다.
    term_token_seqs = []
    for t in terms:
        tnorm = normalize_text(t)
        term_token_seqs.append({
            tuple(tok(" " + tnorm, add_special_tokens=False).input_ids),  # mid-sentence
            tuple(tok(tnorm, add_special_tokens=False).input_ids),         # sentence-initial
        })

    ranks = []
    for row in test:
        ref = normalize_text(row["text"])
        if not any(t in ref for t in terms):
            continue
        feats = proc.feature_extractor(audio_to_array(row),
                sampling_rate=int(row["sampling_rate"]), return_tensors="pt"
                ).input_features.to("cuda", dtype=torch.float16)
        labels = tok(text_target=ref, return_tensors="pt").input_ids.to("cuda")  # 특수 prefix 포함
        with torch.no_grad():
            logits = model(input_features=feats, decoder_input_ids=labels[:, :-1]).logits[0]
        tgt = [int(x) for x in labels[0, 1:].tolist()]
        # 각 용어 토큰열이 tgt의 부분열로 나타나는 위치에서, 첫 subword의 rank를 기록
        for seqs in term_token_seqs:
            for seq in seqs:
                L = len(seq)
                for i in range(len(tgt) - L + 1):
                    if tuple(tgt[i:i + L]) == seq:
                        first_id = seq[0]
                        rank = int((logits[i] > logits[i, first_id]).sum()) + 1  # 1-based
                        ranks.append(rank)
    ranks.sort()
    buckets = {"<=8": 0, "9-16": 0, "17-64": 0, ">64": 0}
    for r in ranks:
        k = "<=8" if r <= 8 else "9-16" if r <= 16 else "17-64" if r <= 64 else ">64"
        buckets[k] += 1
    out = {"n": len(ranks), "buckets": buckets}
    os.makedirs(cfg["paths"]["data_dir"], exist_ok=True)
    json.dump(out, open(os.path.join(cfg["paths"]["data_dir"], "rank_cdf.json"), "w"),
              ensure_ascii=False, indent=2)
    print(out)


if __name__ == "__main__":
    main()
