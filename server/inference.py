import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoTokenizer, AutoConfig
from huggingface_hub import hf_hub_download
from collections import OrderedDict
import onnxruntime as ort
import numpy as np
import os

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
    def __init__(
        self,
        repo_id="DrDrunkenstein22/mbart-kn-en-finetune",
        device="cuda",
        use_onnx=True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_onnx = use_onnx
        self.onnx_session = None
        self.onnx_model_path = "translation_model.onnx"

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
        self.config = config

        weights_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")

        # Load model weights
        checkpoint = torch.load(weights_path, map_location=self.device)

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            # If checkpoint contains nested state dict
            if isinstance(checkpoint["model_state_dict"], str):
                # If it's a path, load from that path
                state_dict = torch.load(
                    checkpoint["model_state_dict"], map_location="cpu"
                )["state_dict"]
            else:
                # If it's already the state dict
                state_dict = checkpoint["model_state_dict"]
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
        else:
            # Direct state dict in checkpoint
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()
        print("PyTorch model loaded")

        # Export and setup ONNX if requested
        if self.use_onnx:
            self._setup_onnx_model()

        print("Model initialization completed")

    def _setup_onnx_model(self):
        """Setup ONNX model for optimized inference"""
        try:
            if not os.path.exists(self.onnx_model_path):
                print("Exporting model to ONNX format...")
                self._export_to_onnx()

            print("Loading ONNX model...")
            providers = ["CPUExecutionProvider"]
            if self.device.type == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

            self.onnx_session = ort.InferenceSession(
                self.onnx_model_path, providers=providers
            )
            print("ONNX model loaded successfully")

        except Exception as e:
            print(f"Failed to setup ONNX model: {e}")
            print("Falling back to PyTorch model")
            self.use_onnx = False
            self.onnx_session = None

    def _export_to_onnx(self):
        """Export PyTorch model to ONNX format"""
        # Create sample inputs for ONNX export
        sample_text = "ಹಲೋ ವರ್ಲ್ಡ್"  # Sample Kannada text
        sample_inputs = self.tokenizer(
            sample_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )

        src_ids = sample_inputs.input_ids.to(self.device)
        # Create dummy target input for decoder
        tgt_ids = torch.tensor([[self.tokenizer.bos_token_id]], device=self.device)

        # Export to ONNX
        torch.onnx.export(
            self.model,
            (src_ids, tgt_ids),
            self.onnx_model_path,
            export_params=True,
            opset_version=15,
            input_names=["src_ids", "tgt_ids"],
            output_names=["logits"],
            dynamic_axes={
                "src_ids": {0: "batch_size", 1: "src_seq_len"},
                "tgt_ids": {0: "batch_size", 1: "tgt_seq_len"},
                "logits": {0: "batch_size", 1: "tgt_seq_len"},
            },
            do_constant_folding=True,
        )
        print(f"Model exported to {self.onnx_model_path}")

    @torch.no_grad()
    def translate(self, text: str, max_length: int = 128, **kwargs):
        """
        Translate text using either ONNX or PyTorch model
        """
        if self.use_onnx and self.onnx_session is not None:
            return self._translate_onnx(text, max_length)
        else:
            return self._translate_pytorch(text, max_length)

    def _translate_pytorch(self, text: str, max_length: int = 128):
        """Translate using PyTorch model"""
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

    def _translate_onnx(self, text: str, max_length: int = 128):
        """Translate using ONNX model with greedy decoding"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        src_ids = inputs.input_ids.numpy().astype(np.int64)

        # Initialize decoder with BOS token
        decoder_input_ids = np.array([[self.tokenizer.bos_token_id]], dtype=np.int64)

        # Greedy decoding
        for _ in range(max_length):
            # Run ONNX inference
            onnx_inputs = {"src_ids": src_ids, "tgt_ids": decoder_input_ids}

            logits = self.onnx_session.run(["logits"], onnx_inputs)[0]

            # Get next token (greedy)
            next_token_id = np.argmax(logits[0, -1, :]).astype(np.int64)

            # Stop if EOS token
            if next_token_id == self.tokenizer.eos_token_id:
                break

            # Append next token
            decoder_input_ids = np.concatenate(
                [decoder_input_ids, [[next_token_id]]], axis=1
            )

        # Decode to text
        translation = self.tokenizer.decode(
            decoder_input_ids[0], skip_special_tokens=True
        )
        return translation
                