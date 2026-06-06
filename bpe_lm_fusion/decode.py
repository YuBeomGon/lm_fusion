# bpe_lm_fusion/decode.py
"""Whisper decoding with optional BPE-LM fusion. Returns 1-best + n-best."""
from __future__ import annotations
import torch
from transformers import (WhisperForConditionalGeneration, WhisperProcessor,
                          LogitsProcessorList)
from .fusion_processor import BpeKenlmFusionProcessor
from .data import audio_to_array


class FusionDecoder:
    def __init__(self, model_name: str, language: str, task: str,
                 device: str = "cuda", dtype: torch.dtype = torch.float16):
        # fp16 by default: standard for Whisper inference and fits the 16GB GPU
        # even when other processes hold VRAM. Pass dtype=torch.float32 to override.
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype).to(device)
        self.model.eval()
        self.device = device
        self.dtype = dtype
        self.language, self.task = language, task
        # skip set = special ids + timestamp ids (timestamps are NOT in all_special_ids)
        tok = self.processor.tokenizer
        skip = set(tok.all_special_ids)
        ts_begin = tok.convert_tokens_to_ids("<|0.00|>")
        if isinstance(ts_begin, int) and ts_begin > 0:
            skip |= set(range(ts_begin, self.model.config.vocab_size))
        self.skip_ids = skip

    def _features(self, row):
        wav = audio_to_array(row)
        feats = self.processor.feature_extractor(
            wav, sampling_rate=int(row["sampling_rate"]), return_tensors="pt"
        ).input_features
        return feats.to(self.device, dtype=self.dtype)

    @torch.no_grad()
    def transcribe(self, row, scorer=None, alpha=0.0, asr_topk=50,
                   mode="topk", num_beams=5, n_best=5):
        feats = self._features(row)
        procs = LogitsProcessorList()
        if scorer is not None and alpha > 0.0:
            procs.append(BpeKenlmFusionProcessor(
                scorer=scorer, alpha=alpha, asr_topk=asr_topk,
                skip_ids=self.skip_ids, mode=mode))
        gen = self.model.generate(
            feats, language=self.language, task=self.task,
            num_beams=num_beams, num_return_sequences=min(n_best, num_beams),
            logits_processor=procs, return_dict_in_generate=True)
        seqs = gen.sequences
        texts = self.processor.batch_decode(seqs, skip_special_tokens=True)
        return {"best": texts[0], "nbest": texts}
