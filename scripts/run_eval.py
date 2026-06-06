# scripts/run_eval.py
"""Run baseline vs fusion over alpha grid for honest & ceiling LMs.

Usage: python scripts/run_eval.py --config configs/poc.yaml --lm-cond honest
출력: data/results_<cond>.jsonl  (alpha별 지표 한 줄)
"""
import argparse, json, os, yaml
from bpe_lm_fusion.data import load_dataset, audio_to_array  # noqa
from bpe_lm_fusion.decode import FusionDecoder
from bpe_lm_fusion.kenlm_scorer import KenlmScorer
from bpe_lm_fusion.domain_terms import load_terms
from bpe_lm_fusion.normalize import normalize_text
from bpe_lm_fusion import metrics, oracle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/poc.yaml")
    ap.add_argument("--lm-cond", choices=["honest", "ceiling"], default="honest")
    ap.add_argument("--limit", type=int, default=0, help="0=full test set")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    ds = load_dataset(cfg["dataset_path"])
    test = ds[cfg["test_split"]]
    if args.limit:
        test = test.select(range(args.limit))
    terms = load_terms(cfg["domain_terms_file"])
    # known Whisper hallucination phrases (config-driven; empty -> metric = 0.0)
    halluc_phrases = cfg.get("hallucination_phrases", [])

    dec = FusionDecoder(cfg["model_name"], cfg["language"], cfg["task"])
    lm_path = os.path.join(cfg["paths"]["data_dir"], f"lm_{args.lm_cond}_5g.binary")
    scorer = KenlmScorer(lm_path)

    refs = [normalize_text(r["text"]) for r in test]
    out_path = os.path.join(cfg["paths"]["data_dir"], f"results_{args.lm_cond}.jsonl")
    with open(out_path, "w") as fout:
        n_total = len(test)
        for alpha in cfg["fusion"]["alpha_grid"]:
            print(f"[{args.lm_cond}] alpha={alpha} start ({n_total} samples)",
                  flush=True)
            hyps, nbests = [], []
            for i, row in enumerate(test, 1):
                res = dec.transcribe(
                    row, scorer=scorer, alpha=alpha,
                    asr_topk=cfg["fusion"]["asr_topk"], mode=cfg["fusion"]["mode"],
                    num_beams=5, n_best=5)
                hyps.append(normalize_text(res["best"]))
                nbests.append([normalize_text(t) for t in res["nbest"]])
                if i % 50 == 0 or i == n_total:
                    print(f"[{args.lm_cond}] alpha={alpha} {i}/{n_total}",
                          flush=True)
            rec = {
                "lm_cond": args.lm_cond, "alpha": alpha,
                "cer": metrics.cer(refs, hyps), "wer": metrics.wer(refs, hyps),
                "insertion_rate": metrics.insertion_rate(refs, hyps),
                # 안정성(환각) 지표 — α 키울 때 게이트 판정용 (가이드 §20/§22)
                "repeated_text_rate": metrics.repeated_text_rate(hyps),
                "no_speech_halluc_rate":
                    metrics.no_speech_hallucination_rate(refs, hyps),
                "halluc_phrase_hit_rate":
                    metrics.hallucination_phrase_hit_rate(hyps, halluc_phrases),
                **{f"length_{k}": v for k, v in
                   metrics.length_ratio_stats(refs, hyps).items()},
                **{f"term_{k}": v for k, v in
                   metrics.term_recall_precision(refs, hyps, terms).items()},
                **{f"oracle_{k}": v for k, v in
                   oracle.nbest_oracle_term_recall(refs, nbests, terms).items()},
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
            print(rec)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
