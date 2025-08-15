import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
from transformers import AutoTokenizer, AutoConfig
from huggingface_hub import hf_hub_download
from collections import OrderedDict

# same model arch class instance 

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._cached_cos = {}
        self._cached_sin = {}

    def forward(self, seq_len, device):
        cache_key = (seq_len, device.type)
        if cache_key in self._cached_cos:
            return self._cached_cos[cache_key], self._cached_sin[cache_key]

        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_emb = emb.cos()[None, None, :, :]
        sin_emb = emb.sin()[None, None, :, :]

        if len(self._cached_cos) < 10:
            self._cached_cos[cache_key] = cos_emb
            self._cached_sin[cache_key] = sin_emb

        return cos_emb, sin_emb


def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = dropout

    def forward(self, x, cos, sin, mask=None, is_causal=False):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = dropout

    def forward(self, x, encoder_output, mask=None):
        B, T, C = x.shape
        _, S, _ = encoder_output.shape

        q = self.query(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = (
            self.key(encoder_output)
            .view(B, S, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.value(encoder_output)
            .view(B, S, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout_p if self.training else 0.0
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class GLU(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.w1(x))
        linear = self.w3(x)
        return self.w2(self.dropout(gate * linear))


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = SelfAttention(d_model, n_heads, dropout)
        self.feed_forward = GLU(d_model, d_ff, dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x, cos, sin, mask=None):
        x = x + self.attention(self.norm1(x), cos, sin, mask=mask, is_causal=False)
        x = x + self.feed_forward(self.norm2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = SelfAttention(d_model, n_heads, dropout)
        self.cross_attention = CrossAttention(d_model, n_heads, dropout)
        self.feed_forward = GLU(d_model, d_ff, dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)

    def forward(self, x, encoder_output, cos, sin, src_mask, tgt_mask):
        x = x + self.self_attention(
            self.norm1(x), cos, sin, mask=tgt_mask, is_causal=True
        )
        x = x + self.cross_attention(self.norm2(x), encoder_output, mask=src_mask)
        x = x + self.feed_forward(self.norm3(x))
        return x


class Seq2SeqTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )
        self.rotary_emb = RotaryPositionalEmbedding(
            config.d_model // config.n_heads, config.max_length
        )

        self.encoder_layers = nn.ModuleList(
            [
                EncoderBlock(
                    config.d_model, config.n_heads, config.d_ff, config.dropout
                )
                for _ in range(config.n_encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                DecoderBlock(
                    config.d_model, config.n_heads, config.d_ff, config.dropout
                )
                for _ in range(config.n_decoder_layers)
            ]
        )

        self.output_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None):
        src_len = src_ids.shape[1]
        device = src_ids.device
        src_emb = self.embedding(src_ids) * math.sqrt(self.d_model)
        cos_src, sin_src = self.rotary_emb(src_len, device)

        encoder_output = src_emb
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, cos_src, sin_src, mask=src_mask)

        tgt_len = tgt_ids.shape[1]
        tgt_emb = self.embedding(tgt_ids) * math.sqrt(self.d_model)
        cos_tgt, sin_tgt = self.rotary_emb(tgt_len, device)

        decoder_output = tgt_emb
        for layer in self.decoder_layers:
            decoder_output = layer(
                decoder_output, encoder_output, cos_tgt, sin_tgt, src_mask, tgt_mask
            )

        decoder_output = self.output_norm(decoder_output)
        logits = self.lm_head(decoder_output)
        return logits

    @torch.no_grad()
    def generate(self, src_ids, tokenizer, max_new_tokens=50):
        self.eval()
        device = src_ids.device
        decoder_input_ids = torch.tensor(
            [[tokenizer.bos_token_id]], dtype=torch.long, device=device
        )

        for _ in range(max_new_tokens):
            logits = self(src_ids, decoder_input_ids)
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1)

            decoder_input_ids = torch.cat(
                [decoder_input_ids, next_token_id.unsqueeze(0)], dim=1
            )

            if next_token_id.item() == tokenizer.eos_token_id:
                break

        return decoder_input_ids


class TranslationInference:
    def __init__(self, repo_id="DrDrunkenstein22/mbart-kn-en-finetune", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        print(f"Loading tokenizer and model from '{repo_id}'...")

        self.tokenizer = AutoTokenizer.from_pretrained(repo_id)
        config = AutoConfig.from_pretrained(repo_id)

        
        config.vocab_size = len(self.tokenizer)

        # set bos, eos token id
        if not hasattr(config, "bos_token_id") or config.bos_token_id is None:
            config.bos_token_id = self.tokenizer.bos_token_id
        if not hasattr(config, "eos_token_id") or config.eos_token_id is None:
            config.eos_token_id = self.tokenizer.eos_token_id

        self.model = Seq2SeqTransformer(config)

        weights_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")

        # fetching state dict
        checkpoint = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.to(self.device)
        self.model.eval()
        print("model loaded")

    @torch.no_grad()
    def translate(self, text: str, max_length: int = 128, **kwargs):
        """
        Beam search optimization for inference ? 
        """
        self.model.eval()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = inputs.input_ids.to(self.device)

        output_ids = self.model.generate(
            input_ids, self.tokenizer, max_new_tokens=max_length
        )
        translation = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return translation


