"""Llama-style causal language model builder for T2T² experiments."""

from typing import Dict

from transformers import LlamaConfig, LlamaForCausalLM


def build_llama_model(config: Dict) -> LlamaForCausalLM:
    """Build a Llama-style causal LM from a model config dict.

    Args:
        config: Dict with keys n_layers, hidden_dim, n_heads, intermediate_dim,
                max_seq_len, vocab_size, tie_embeddings.

    Returns:
        Initialized LlamaForCausalLM instance (randomly initialized weights).
    """
    # The config's intermediate_dim follows the "4× hidden" convention.
    # Llama's SwiGLU uses 3 weight matrices, so actual intermediate = 2/3 × that,
    # keeping total FFN params equivalent to a standard 2-matrix FFN.
    actual_intermediate = int(2 / 3 * config["intermediate_dim"])

    llama_cfg = LlamaConfig(
        hidden_size=config["hidden_dim"],
        intermediate_size=actual_intermediate,
        num_hidden_layers=config["n_layers"],
        num_attention_heads=config["n_heads"],
        num_key_value_heads=config["n_heads"],
        max_position_embeddings=config["max_seq_len"],
        vocab_size=config["vocab_size"],
        tie_word_embeddings=config.get("tie_embeddings", True),
        hidden_act="silu",
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        attention_bias=False,
        bos_token_id=50256,
        eos_token_id=50256,
    )
    model = LlamaForCausalLM(llama_cfg)
    return model


def count_parameters(model: LlamaForCausalLM) -> Dict[str, int]:
    """Count total, non-embedding, and embedding parameters.

    Args:
        model: A LlamaForCausalLM instance.

    Returns:
        Dict with keys 'total', 'non_embedding', 'embedding'.
    """
    total = sum(p.numel() for p in model.parameters())

    vocab_size = model.config.vocab_size
    hidden_dim = model.config.hidden_size

    # For tied embeddings: embedding table is shared between input & output,
    # so embedding params = vocab_size × hidden_dim (counted once).
    # Without tying: 2 × vocab_size × hidden_dim.
    if model.config.tie_word_embeddings:
        embedding_params = vocab_size * hidden_dim
    else:
        embedding_params = 2 * vocab_size * hidden_dim

    non_embedding = total - embedding_params

    return {
        "total": total,
        "non_embedding": non_embedding,
        "embedding": embedding_params,
    }
