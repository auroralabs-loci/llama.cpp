#!/usr/bin/env python3
"""
Convert Reka Yasa2 checkpoints to GGUF (text decoder and optional mmproj for MTMD).

- Text: Llama-arch decoder + tiktoken/BPE vocab (bytes keys normalized for GGUF).
- Mmproj (--mmproj): ConvNeXt vision + language_projection for Yasa2 MTMD path.
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import torch
from transformers import AutoTokenizer

import convert_hf_to_gguf as base


def _get_2d_sincos_pos_embed_yasa2(embed_dim: int, image_size: int = 50, seq_len: int = 256) -> np.ndarray:
    """Match HF get_2d_sincos_pos_embed(hidden, image_size=50) then slice [:seq_len]."""
    assert embed_dim % 2 == 0

    def _get_1d(embed_dim_1d: int, pos_2d: np.ndarray) -> np.ndarray:
        assert embed_dim_1d % 2 == 0
        omega = np.arange(embed_dim_1d // 2, dtype=np.float32)
        omega /= embed_dim_1d / 2.0
        omega = 1.0 / (10000.0**omega)
        out = np.einsum("hw,d->hwd", pos_2d, omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=-1)

    grid_h = np.arange(image_size, dtype=np.float32)
    grid_w = np.arange(image_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    emb_h = _get_1d(embed_dim // 2, grid[0])
    emb_w = _get_1d(embed_dim // 2, grid[1])
    pos = np.concatenate([emb_h, emb_w], axis=-1).reshape(image_size * image_size, embed_dim)
    return pos[:seq_len].astype(np.float32, copy=False)


class RekaYasa2TextDecoderModel(base.LlamaModel):
    model_arch = base.gguf.MODEL_ARCH.LLAMA

    def get_vocab_base(self) -> tuple[list[str], list[int], str]:
        tokens: list[str] = []
        toktypes: list[int] = []

        tokenizer = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=True)
        vocab_dict = tokenizer.get_vocab()
        vocab_size = self.hparams.get("vocab_size", len(vocab_dict))
        assert max(vocab_dict.values()) < vocab_size

        tokpre = self.get_vocab_base_pre(tokenizer)
        reverse_vocab = {idx: tok for tok, idx in vocab_dict.items()}
        added_vocab = tokenizer.get_added_vocab()
        added_tokens_decoder = getattr(tokenizer, "added_tokens_decoder", {})
        tiktoken_special = set(getattr(tokenizer, "tiktoken_special_tokens", {}).keys())

        def token_to_str(tok: str | bytes) -> str:
            if isinstance(tok, bytes):
                return base.QwenModel.token_bytes_to_string(tok)
            return tok

        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(base.gguf.TokenType.UNUSED)
                continue

            token = reverse_vocab[i]
            token_str = token_to_str(token)
            if token_str in tiktoken_special:
                toktypes.append(base.gguf.TokenType.CONTROL)
            elif token in added_vocab or token_str in added_vocab:
                if i in added_tokens_decoder and getattr(added_tokens_decoder[i], "special", False):
                    toktypes.append(base.gguf.TokenType.CONTROL)
                else:
                    toktypes.append(base.gguf.TokenType.USER_DEFINED)
            else:
                toktypes.append(base.gguf.TokenType.NORMAL)
            tokens.append(token_str)

        return tokens, toktypes, tokpre

    def set_vocab(self):
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        tokenizer = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=True)
        mergeable_ranks = getattr(tokenizer, "mergeable_ranks", None)
        if mergeable_ranks is None and hasattr(tokenizer, "tiktoken"):
            mergeable_ranks = getattr(tokenizer.tiktoken, "_mergeable_ranks", None)

        if mergeable_ranks:
            merges: list[str] = []
            for token, rank in mergeable_ranks.items():
                if len(token) == 1:
                    continue
                merged = base.QwenModel.bpe(mergeable_ranks, token, max_rank=rank)
                if len(merged) == 2:
                    merges.append(" ".join(map(base.QwenModel.token_bytes_to_string, merged)))
            self.gguf_writer.add_token_merges(merges)

        special_vocab = base.gguf.SpecialVocab(self.dir_model, load_merges=False)
        special_vocab.add_to_gguf(self.gguf_writer)

        # SpecialVocab finds no eos_token in the tokenizer config and writes nothing,
        # causing llama.cpp to fall back to the gpt2 default of token 11 (comma) as EOS.
        # Explicitly write the correct values: <|endoftext|> as EOS, <sep> as EOT.
        self.gguf_writer.add_eos_token_id(tokenizer.tiktoken.encode_single_token("<|endoftext|>"))
        self.gguf_writer.add_eot_token_id(tokenizer.tiktoken.encode_single_token("<sep>"))

        # Tiktoken/BPE typically does not add BOS; avoid shifting logits.
        self.gguf_writer.add_add_bos_token(False)

    def modify_tensors(
        self, data_torch, name: str, bid: int | None
    ) -> Iterable[tuple[str, object]]:
        if name.startswith("model.language_model."):
            name = "model." + name[len("model.language_model.") :]
        elif name.startswith("language_model."):
            name = name[len("language_model.") :]
        elif name.startswith("model.vision_model.") or name.startswith("model.connector."):
            return
        elif name.startswith("vision_model.") or name.startswith("connector."):
            return

        if not (
            name == "lm_head.weight"
            or name == "model.embed_tokens.weight"
            or name == "model.norm.weight"
            or name.startswith("model.layers.")
        ):
            return

        yield from super().modify_tensors(data_torch, name, bid)


class RekaYasa2VisionMmprojModel(base.MmprojModel):
    """Vision backbone + language_projection tensors for MTMD Yasa2 clip graph."""

    model_arch = base.gguf.MODEL_ARCH.MMPROJ
    has_vision_encoder = True
    has_audio_encoder = False

    def get_vision_config(self) -> dict[str, object] | None:
        cfg = self.global_config.get("vision_config")
        if not isinstance(cfg, dict):
            return cfg

        out = dict(cfg)
        depths = out.get("depths")
        if isinstance(depths, list) and "num_hidden_layers" not in out:
            out["num_hidden_layers"] = int(sum(int(x) for x in depths))
        return out

    def set_gguf_parameters(self):
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_clip_has_vision_encoder(True)
        self.gguf_writer.add_clip_projector_type("yasa2")
        self.gguf_writer.add_vision_projection_dim(self.n_embd_text)

        vcfg = self.global_config.get("vision_config", {})
        self.gguf_writer.add_vision_image_size(int(vcfg.get("image_size", 512)))
        self.gguf_writer.add_vision_patch_size(int(vcfg.get("patch_size", 4)))
        self.gguf_writer.add_vision_embedding_length(int(vcfg.get("hidden_size", 2816)))
        self.gguf_writer.add_vision_feed_forward_length(int(vcfg.get("hidden_size", 2816)))
        self.gguf_writer.add_vision_block_count(0)
        self.gguf_writer.add_vision_head_count(int(vcfg.get("num_attention_heads", 1) or 1))
        self.gguf_writer.add_vision_attention_layernorm_eps(float(vcfg.get("layer_norm_eps", 1e-6)))
        self.gguf_writer.add_vision_use_gelu(True)

        mean = self.preprocessor_config.get("image_mean", [0.485, 0.456, 0.406])
        std = self.preprocessor_config.get("image_std", [0.229, 0.224, 0.225])
        self.gguf_writer.add_vision_image_mean(mean)
        self.gguf_writer.add_vision_image_std(std)
        use_vision_pos = self.global_config.get("use_vision_pos_embed", True)
        if use_vision_pos:
            hidden = int(vcfg.get("hidden_size", 2816))
            self._yasa_vision_pos_embed = _get_2d_sincos_pos_embed_yasa2(hidden, 50, 256)
        else:
            self._yasa_vision_pos_embed = None
        self._yasa_pos_embed_yielded = False

    def _ensure_yasa_pos_embed(self) -> None:
        if hasattr(self, "_yasa_vision_pos_embed"):
            return
        self._yasa_pos_embed_yielded = False
        vcfg = self.global_config.get("vision_config", {})
        use_vision_pos = self.global_config.get("use_vision_pos_embed", True)
        if use_vision_pos:
            hidden = int(vcfg.get("hidden_size", 2816))
            self._yasa_vision_pos_embed = _get_2d_sincos_pos_embed_yasa2(hidden, 50, 256)
        else:
            self._yasa_vision_pos_embed = None

    def modify_tensors(
        self, data_torch, name: str, bid: int | None
    ) -> Iterable[tuple[str, object]]:
        del bid

        if name.startswith("model.vision_model.") or name.startswith("model.language_projection."):
            self._ensure_yasa_pos_embed()
            short = name[len("model.") :]
            if not self._yasa_pos_embed_yielded and self._yasa_vision_pos_embed is not None and "vision_model." in short:
                self._yasa_pos_embed_yielded = True
                yield "v.vision_pos_embed", torch.from_numpy(self._yasa_vision_pos_embed.copy())
            out_name = self._map_mmproj_name(short)
            if out_name is not None:
                yield out_name, data_torch
            return

        if name.startswith("vision_model.") or name.startswith("language_projection."):
            self._ensure_yasa_pos_embed()
            if not self._yasa_pos_embed_yielded and self._yasa_vision_pos_embed is not None and "vision_model." in name:
                self._yasa_pos_embed_yielded = True
                yield "v.vision_pos_embed", torch.from_numpy(self._yasa_vision_pos_embed.copy())
            out_name = self._map_mmproj_name(name)
            if out_name is not None:
                yield out_name, data_torch
            return

        return

    @staticmethod
    def _map_mmproj_name(name: str) -> str | None:
        if name.startswith("language_projection."):
            mapping = {
                "language_projection.0.weight": "mm.0.weight",
                "language_projection.0.bias": "mm.0.bias",
                "language_projection.2.weight": "mm.2.weight",
                "language_projection.2.bias": "mm.2.bias",
            }
            return mapping.get(name)

        simple = {
            "vision_model.backbone.embeddings.patch_embeddings.weight": "v.patch_embd.weight",
            "vision_model.backbone.embeddings.patch_embeddings.bias": "v.patch_embd.bias",
            "vision_model.backbone.embeddings.layernorm.weight": "v.patch_ln.weight",
            "vision_model.backbone.embeddings.layernorm.bias": "v.patch_ln.bias",
            "vision_model.backbone.layernorm.weight": "v.backbone_ln.weight",
            "vision_model.backbone.layernorm.bias": "v.backbone_ln.bias",
        }
        if name in simple:
            return simple[name]

        m = re.match(
            r"vision_model\.backbone\.encoder\.stages\.(\d+)\.downsampling_layer\.(0|1)\.(weight|bias)$",
            name,
        )
        if m:
            stage = int(m.group(1))
            layer_idx = int(m.group(2))
            wb = m.group(3)
            if layer_idx == 0:
                return f"v.stage.{stage}.down.ln.{wb}"
            return f"v.stage.{stage}.down.conv.{wb}"

        m = re.match(
            r"vision_model\.backbone\.encoder\.stages\.(\d+)\.layers\.(\d+)\.(dwconv|layernorm|pwconv1|grn|pwconv2)\.(weight|bias)$",
            name,
        )
        if m:
            stage = int(m.group(1))
            blk = int(m.group(2))
            part = m.group(3)
            wb = m.group(4)
            part_map = {
                "dwconv": "dw",
                "layernorm": "ln",
                "pwconv1": "pw1",
                "grn": "grn",
                "pwconv2": "pw2",
            }
            return f"v.stage.{stage}.blk.{blk}.{part_map[part]}.{wb}"

        return None

    def tensor_force_quant(
        self, name: str, new_name: str, bid: int | None, n_dims: int
    ) -> base.gguf.GGMLQuantizationType | bool:
        del name, new_name, bid
        if n_dims > 1:
            return base.gguf.GGMLQuantizationType.F16
        return False


def register_reka_architectures() -> None:
    model_classes = base.ModelBase._model_classes[base.ModelType.TEXT]
    mmproj_classes = base.ModelBase._model_classes[base.ModelType.MMPROJ]
    for arch_name in (
        "Yasa2ForConditionalGeneration",
        "Yasa2Model",
        "YasaCausalLM",
    ):
        model_classes[arch_name] = RekaYasa2TextDecoderModel
    mmproj_classes["Yasa2ForConditionalGeneration"] = RekaYasa2VisionMmprojModel


def main() -> None:
    register_reka_architectures()
    base.main()


if __name__ == "__main__":
    main()
