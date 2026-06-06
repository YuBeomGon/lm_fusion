"""
scripts/write_sweep_report.py
Write a Markdown report for top-k/alpha sweep JSONL outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_topks(raw: str) -> list[int]:
    return [int(x) for x in raw.split() if x.strip()]


def _parse_alphas(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _load_records(data_dir: Path, lm_cond: str, mode: str,
                  topks: list[int], run_id: str) -> list[dict]:
    records: list[dict] = []
    for topk in topks:
        path = data_dir / f"results_{lm_cond}_sweep_{mode}_topk{topk}_{run_id}.jsonl"
        with path.open() as f:
            for line in f:
                rec = json.loads(line)
                rec["topk"] = topk
                rec["source_file"] = str(path)
                records.append(rec)
    return records


def _delta(rec: dict, baseline: dict | None, key: str) -> str:
    if baseline is None:
        return "-"
    return f"{rec[key] - baseline[key]:+.4f}"


def _best(records: list[dict]) -> dict:
    safe = [
        rec for rec in records
        if rec.get("repeated_text_rate", 0.0) == 0.0
        and rec.get("no_speech_halluc_rate", 0.0) == 0.0
        and rec.get("halluc_phrase_hit_rate", 0.0) == 0.0
    ]
    pool = safe or records
    return min(pool, key=lambda r: (r["cer"], -r["term_recall"], r["topk"], r["alpha"]))


def _write_table(lines: list[str], records: list[dict]) -> None:
    lines.append("| topk | alpha | CER | ΔCER | WER | term recall | Δrecall | precision | sec/sample | samples/sec | elapsed min | insertion | repeated | skipped long |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    baseline_records = [r for r in records if r["alpha"] == 0.0]
    common_baseline = baseline_records[0] if baseline_records else None
    baselines = {r["topk"]: r for r in baseline_records}
    for rec in sorted(records, key=lambda r: (r["topk"], r["alpha"])):
        baseline = baselines.get(rec["topk"], common_baseline)
        drec = "-" if baseline is None else f"{(rec['term_recall'] - baseline['term_recall']) * 100:+.1f}%p"
        sec_per_sample = rec.get("seconds_per_sample", 0.0)
        samples_per_second = rec.get("samples_per_second", 0.0)
        elapsed_min = rec.get("elapsed_seconds", 0.0) / 60.0
        lines.append(
            f"| {rec['topk']} | {rec['alpha']:.2f} | {rec['cer']:.4f} | "
            f"{_delta(rec, baseline, 'cer')} | {rec['wer']:.4f} | "
            f"{rec['term_recall']:.4f} | {drec} | {rec['term_precision']:.4f} | "
            f"{sec_per_sample:.3f} | {samples_per_second:.3f} | {elapsed_min:.1f} | "
            f"{rec['insertion_rate']:.4f} | {rec['repeated_text_rate']:.4f} | "
            f"{rec.get('skipped_long_audio', 0)} |"
        )


def _write_speed_summary(lines: list[str], records: list[dict]) -> None:
    lines.append("| topk | mean sec/sample | mean samples/sec | total elapsed min |")
    lines.append("|---:|---:|---:|---:|")
    topks = sorted({rec["topk"] for rec in records})
    for topk in topks:
        group = [rec for rec in records if rec["topk"] == topk]
        mean_sps = sum(rec.get("seconds_per_sample", 0.0) for rec in group) / len(group)
        mean_tput = sum(rec.get("samples_per_second", 0.0) for rec in group) / len(group)
        total_min = sum(rec.get("elapsed_seconds", 0.0) for rec in group) / 60.0
        lines.append(f"| {topk} | {mean_sps:.3f} | {mean_tput:.3f} | {total_min:.1f} |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--lm-cond", default="honest")
    parser.add_argument("--mode", default="topk_strict")
    parser.add_argument("--topks", required=True)
    parser.add_argument("--alphas", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    topks = _parse_topks(args.topks)
    alphas = _parse_alphas(args.alphas)
    records = _load_records(Path(args.data_dir), args.lm_cond, args.mode,
                            topks, args.run_id)
    best = _best(records)

    lines = [
        f"# Top-k / Alpha Sweep Report — {args.run_id}",
        "",
        "## 조건",
        "",
        f"- LM condition: `{args.lm_cond}`",
        f"- fusion mode: `{args.mode}`",
        f"- top-k grid: `{', '.join(str(x) for x in topks)}`",
        f"- alpha grid: `{', '.join(f'{x:.2f}' for x in alphas)}`",
        "- dataset: full validation/test set after `max_audio_seconds` filter",
        "- term metric: longest-match, non-overlapping, whitespace-insensitive",
        "",
        "## 요약",
        "",
        f"- Best safe CER: topk={best['topk']}, alpha={best['alpha']:.2f}, "
        f"CER={best['cer']:.4f}, WER={best['wer']:.4f}, "
        f"term recall={best['term_recall']:.4f}, precision={best['term_precision']:.4f}",
        "- 기존 POC 결과와 직접 비교할 때는 metric과 fusion mode가 바뀐 점을 반영해야 한다.",
        "",
        "## 속도 요약",
        "",
    ]
    _write_speed_summary(lines, records)
    lines.extend([
        "",
        "## 전체 결과",
        "",
    ])
    _write_table(lines, records)
    lines.extend([
        "",
        "## 산출물",
        "",
    ])
    for topk in topks:
        lines.append(
            f"- `data/results_{args.lm_cond}_sweep_{args.mode}_topk{topk}_{args.run_id}.jsonl`"
        )
    lines.append("")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
