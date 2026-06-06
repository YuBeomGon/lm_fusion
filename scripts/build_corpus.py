# scripts/build_corpus.py
"""Write KenLM training corpora: honest (train) and ceiling (train+test).

Usage: python scripts/build_corpus.py --config configs/poc.yaml
"""
import argparse, os, yaml
from transformers import WhisperTokenizer
from bpe_lm_fusion.data import load_dataset, split_texts
from bpe_lm_fusion.corpus import build_corpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/poc.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    tok = WhisperTokenizer.from_pretrained(
        cfg["model_name"], language=cfg["language"], task=cfg["task"])
    ds = load_dataset(cfg["dataset_path"])
    train_txt = split_texts(ds, cfg["train_split"])
    test_txt = split_texts(ds, cfg["test_split"])

    out_dir = cfg["paths"]["data_dir"]
    os.makedirs(out_dir, exist_ok=True)

    honest = build_corpus(train_txt, tok)
    ceiling = build_corpus(train_txt + test_txt, tok)

    for name, lines in [("honest", honest), ("ceiling", ceiling)]:
        path = os.path.join(out_dir, f"corpus_{name}.txt")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"{name}: {len(lines)} lines -> {path}")


if __name__ == "__main__":
    main()
