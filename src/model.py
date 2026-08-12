import torch
import torch.nn as nn
import math
from model_components.TransformerBlock import TransformerBlock
from model_components.RMSNorm import RMSNorm
from model_components.RoPE import RotaryPositionalEmbedding

class TransformerLM(nn.Module):
    """
    A decoder-only Transformer language model in the modern dense style.
    params:
        config: a TransformerConfig
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)

        # One RoPE module shared by every layer: the cos/sin tables are identical everywhere,
        # so there is no reason to store them more than once.
        self.rope = RotaryPositionalEmbedding(
            d_head=config.d_model // config.n_heads,
            max_seq_len=config.context_length,
            theta=config.rope_theta,
        )

        self.layers = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff,
                             self.rope, config.use_qk_norm)
            for _ in range(config.n_layers)
        ])

        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            std = math.sqrt(2.0 / (module.in_features + module.out_features))
            nn.init.trunc_normal_(module.weight, std=std, a=-3 * std, b=3 * std)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids):
        """
        params:
            token_ids: (batch, seq_len) integer token IDs
        returns:
            logits: (batch, seq_len, vocab_size)
        """
        # Step 1: embed the token IDs. No positional embedding - RoPE handles position.
        x = self.token_embeddings(token_ids)              # (batch, seq_len, d_model)

        # Step 2: run the stack of pre-norm blocks
        for layer in self.layers:
            x = layer(x)                                  # (batch, seq_len, d_model)

        # Step 3: final norm, then project to vocabulary logits
        return self.lm_head(self.final_norm(x))           # (batch, seq_len, vocab_size)

    def num_parameters(self, non_embedding=False):
        """Total parameter count; optionally excluding the embedding and LM head."""
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.token_embeddings.weight.numel() + self.lm_head.weight.numel()
        return total