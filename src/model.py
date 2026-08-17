import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from .model_components.TransformerBlock import TransformerBlock
from .model_components.RMSNorm import RMSNorm
from .model_components.RoPE import RotaryPositionalEmbedding

@dataclass
class TransformerConfig:
    vocab_size: int
    context_length: int
    n_layers: int
    d_model: int
    n_heads: int
    d_ff: int
    rope_theta: float = 10000.0
    use_qk_norm: bool = True
    use_rmsnorm: bool = True
    use_rope: bool = True
    ffn_type: str = 'swiglu'   # 'swiglu' or 'relu'

class TransformerLM(nn.Module):
    """
    A decoder-only Transformer language model in the modern dense style.
    params:
        config: a TransformerConfig
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)

        self.rope = RotaryPositionalEmbedding(
            d_head=config.d_model // config.n_heads,
            max_seq_len=config.context_length,
            theta=config.rope_theta,
        )

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                rope=self.rope,
                context_length=config.context_length,
                use_qk_norm=config.use_qk_norm,
                use_rmsnorm=config.use_rmsnorm,
                use_rope=config.use_rope,
                ffn_type=config.ffn_type
            )
            for _ in range(config.n_layers)
        ])

        self.final_norm = RMSNorm(config.d_model) if config.use_rmsnorm else nn.Identity()
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            std = math.sqrt(2.0 / (module.in_features + module.out_features))
            nn.init.trunc_normal_(module.weight, std=std, a=-3 * std, b=3 * std)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids, use_cache=False):
        """
        params:
            token_ids: (batch, seq_len) integer token IDs
        returns:
            logits: (batch, seq_len, vocab_size)
        """
        x = self.token_embeddings(token_ids)  # (batch, seq_len, d_model)

        for layer in self.layers:
            x = layer(x, use_cache=use_cache)

        return self.lm_head(self.final_norm(x))

    def num_parameters(self, non_embedding=False):
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.token_embeddings.weight.numel() + self.lm_head.weight.numel()
        return total