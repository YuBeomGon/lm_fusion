"""
scripts/run_eval.py
Run baseline vs fusion over alpha grid for honest & ceiling LMs.

Usage: python scripts/run_eval.py --config configs/poc.yaml --lm-cond honest
출력: data/results_<cond>.jsonl  (alpha별 지표 한 줄)
"""
import argparse, json, os, yaml
from time import perf_counter
from bpe_lm_fusion.data import load_dataset, audio_duration_seconds
from bpe_lm_fusion.decode import FusionDecoder
from bpe_lm_fusion.kenlm_scorer import KenlmScorer
from bpe_lm_fusion.domain_terms import load_terms
from bpe_lm_fusion.normalize import normalize_text
from bpe_lm_fusion import metrics, oracle


def _filter_by_max_audio_seconds(test, max_audio_seconds):
    if not max_audio_seconds or max_audio_seconds <= 0:
        return test, 0
    keep = [
        i for i, row in enumerate(test)
        if audio_duration_seconds(row) <= max_audio_seconds
    ]
    return test.select(keep), len(test) - len(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/poc.yaml")
    ap.add_argument("--lm-cond", choices=["honest", "ceiling"], default="honest")
    ap.add_argument("--limit", type=int, default=0, help="0=full test set")
    ap.add_argument("--alpha-grid", default="", help="comma-separated alpha override")
    ap.add_argument("--asr-topk", type=int, default=0, help="fusion.asr_topk override")
    ap.add_argument("--fusion-mode", default="", help="fusion.mode override")
    ap.add_argument("--output-suffix", default="", help="suffix for results_<cond>*.jsonl")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.alpha_grid:
        cfg["fusion"]["alpha_grid"] = [
            float(x.strip()) for x in args.alpha_grid.split(",") if x.strip()
        ]
    if args.asr_topk:
        cfg["fusion"]["asr_topk"] = args.asr_topk
    if args.fusion_mode:
        cfg["fusion"]["mode"] = args.fusion_mode

    ds = load_dataset(cfg["dataset_path"])
    test = ds[cfg["test_split"]]
    test, skipped_long = _filter_by_max_audio_seconds(
        test, cfg.get("max_audio_seconds"))
    if skipped_long:
        print(f"[filter] skipped {skipped_long} samples over "
              f"{cfg['max_audio_seconds']}s", flush=True)
    if args.limit:
        test = test.select(range(min(args.limit, len(test))))
    terms = load_terms(cfg["domain_terms_file"])
    # known Whisper hallucination phrases (config-driven; empty -> metric = 0.0)
    halluc_phrases = cfg.get("hallucination_phrases", [])

    decode_cfg = cfg.get("decode", {})
    dec = FusionDecoder(
        cfg["model_name"], cfg["language"], cfg["task"],
        device=decode_cfg.get("device", "cuda"),
        dtype=decode_cfg.get("dtype", "float16"))
    order = int(cfg.get("kenlm", {}).get("order", 5))
    lm_path = os.path.join(
        cfg["paths"]["data_dir"], f"lm_{args.lm_cond}_{order}g.binary")
    needs_lm = any(alpha > 0.0 for alpha in cfg["fusion"]["alpha_grid"])
    scorer = KenlmScorer(lm_path) if needs_lm else None

    refs = [normalize_text(r["text"]) for r in test]
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    out_path = os.path.join(
        cfg["paths"]["data_dir"], f"results_{args.lm_cond}{suffix}.jsonl")
    with open(out_path, "w") as fout:
        n_total = len(test)
        for alpha in cfg["fusion"]["alpha_grid"]:
            alpha_started = perf_counter()
            print(f"[{args.lm_cond}] alpha={alpha} start ({n_total} samples)",
                  flush=True)
            hyps, nbests = [], []
            for i, row in enumerate(test, 1):
                res = dec.transcribe(
                    row, scorer=scorer, alpha=alpha,
                    asr_topk=cfg["fusion"]["asr_topk"], mode=cfg["fusion"]["mode"],
                    num_beams=decode_cfg.get("num_beams", 5),
                    n_best=decode_cfg.get("n_best", 5))
                hyps.append(normalize_text(res["best"]))
                nbests.append([normalize_text(t) for t in res["nbest"]])
                if i % 50 == 0 or i == n_total:
                    print(f"[{args.lm_cond}] alpha={alpha} {i}/{n_total}",
                          flush=True)
            elapsed_seconds = perf_counter() - alpha_started
            samples_per_second = n_total / elapsed_seconds if elapsed_seconds else 0.0
            rec = {
                "lm_cond": args.lm_cond, "alpha": alpha,
                "skipped_long_audio": skipped_long,
                "n_samples": n_total,
                "elapsed_seconds": elapsed_seconds,
                "seconds_per_sample":
                    elapsed_seconds / n_total if n_total else 0.0,
                "samples_per_second": samples_per_second,
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
