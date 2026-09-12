"""Last-Mean readout and training helpers for the final Llama-3 ATV model."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import math
import os

import torch
from torch import Tensor, nn
import torch.nn.functional as F

CHECKPOINT_FORMAT = "atv_lora_dual_readout_v1"
GPT2_LORA_TARGET_MODULES = ("c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")

def _validate_hidden_and_mask(hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    if hidden_state.ndim != 3:
        raise ValueError("hidden_state must have shape [batch, sequence, hidden]")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    if hidden_state.shape[:2] != attention_mask.shape:
        raise ValueError("hidden_state and attention_mask dimensions do not match")
    non_padding = attention_mask.to(device=hidden_state.device, dtype=torch.bool)
    if not bool(non_padding.any(dim=1).all()):
        raise ValueError("every sample must contain at least one non-padding token")
    return non_padding

def select_last_non_padding(hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    """Select the final attention-mask-valid token for every sample."""

    non_padding = _validate_hidden_and_mask(hidden_state, attention_mask)
    positions = torch.arange(
        hidden_state.shape[1], device=hidden_state.device
    ).expand(hidden_state.shape[0], -1)
    last_positions = positions.masked_fill(~non_padding, -1).max(dim=1).values
    batch_indices = torch.arange(hidden_state.shape[0], device=hidden_state.device)
    return hidden_state[batch_indices, last_positions]

def masked_mean_non_padding(hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    """Mean-pool all attention-mask-valid final-layer token states."""

    non_padding = _validate_hidden_and_mask(hidden_state, attention_mask)
    mask = non_padding.to(dtype=hidden_state.dtype).unsqueeze(-1)
    numerator = (hidden_state * mask).sum(dim=1)
    denominator = mask.sum(dim=1)
    return numerator / denominator


class LastMeanProjection(nn.Module):
    """U((1-rho) D_last(z_last) + rho D_mean(z_mean)), without normalization."""

    def __init__(self, input_dim, output_dim, rank=256, mean_alpha=0.7):
        super().__init__()
        if min(input_dim, output_dim, rank) <= 0:
            raise ValueError("projection dimensions must be positive")
        if not math.isfinite(mean_alpha) or not 0 <= mean_alpha <= 1:
            raise ValueError("mean_alpha must be in [0, 1]")
        self.rank = rank
        self.mean_alpha = float(mean_alpha)
        # Preserve the final experiment's initialization order and RNG state.
        self.down_last = nn.Linear(input_dim, rank, bias=True)
        self.up = nn.Linear(rank, output_dim, bias=True)
        with torch.random.fork_rng(devices=[]):
            self.down_mean = nn.Linear(input_dim, rank, bias=False)
        nn.init.zeros_(self.down_mean.weight)

    def forward(self, last, mean):
        return self.up((1.0 - self.mean_alpha) * self.down_last(last)
                       + self.mean_alpha * self.down_mean(mean))


def project_source(gpt2_model, inputs, projection_layer):
    hidden = gpt2_model(**inputs, use_cache=False).last_hidden_state
    last = select_last_non_padding(hidden, inputs["attention_mask"])
    if isinstance(projection_layer, LastMeanProjection):
        mean = masked_mean_non_padding(hidden, inputs["attention_mask"])
        return projection_layer(last, mean)
    return projection_layer(last)

def apply_gpt2_lora(
    gpt2_model: nn.Module,
    *,
    rank: int,
    alpha: int,
    dropout: float,
) -> nn.Module:
    """Freeze GPT-2 and add LoRA to all attention/MLP Conv1D projections."""
    if rank <= 0:
        raise ValueError("LoRA rank must be positive")
    if alpha <= 0:
        raise ValueError("LoRA alpha must be positive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("LoRA dropout must be in [0, 1)")

    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(GPT2_LORA_TARGET_MODULES),
        bias="none",
        fan_in_fan_out=True,
    )
    return get_peft_model(gpt2_model, lora_config)

def _split_named_decay_parameters(named_parameters):
    decay = []
    no_decay = []
    for parameter_name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or parameter_name.endswith("bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return decay, no_decay

def accumulation_group_size(
    sample_index: int,
    *,
    total_samples: int,
    batch_size: int,
) -> int:
    """Return the actual accumulation-group size containing ``sample_index``."""
    if total_samples <= 0:
        raise ValueError("total_samples must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0 <= sample_index < total_samples:
        raise ValueError("sample_index is outside the dataset")
    group_start = (sample_index // batch_size) * batch_size
    return min(batch_size, total_samples - group_start)

def backward_template_losses_memory_efficient(
    *,
    layer_vectors: Tensor,
    template_losses: Iterable[Tensor],
    template_count: int,
    accumulation_group_size: int,
) -> float:
    """Backpropagate lazy template losses while retaining one LLM graph at a time.

    Each yielded loss is differentiated only as far as ``layer_vectors``. Those
    gradients are accumulated, after which one backward call propagates through
    the shared projector/GPT-2 graph. This is mathematically the same objective as
    ``mean(template_losses) / accumulation_group_size``.
    """
    if template_count <= 0:
        raise ValueError("template_count must be positive")
    if accumulation_group_size <= 0:
        raise ValueError("accumulation_group_size must be positive")
    if not layer_vectors.requires_grad:
        raise ValueError("layer_vectors must require gradients")

    total_vector_gradient: Optional[Tensor] = None
    detached_loss_sum: Optional[Tensor] = None
    observed_count = 0
    scale = 1.0 / (template_count * accumulation_group_size)

    for loss in template_losses:
        if loss.ndim != 0:
            raise ValueError("each template loss must be scalar")
        vector_gradient, = torch.autograd.grad(
            loss * scale,
            layer_vectors,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
        vector_gradient = vector_gradient.detach()
        if total_vector_gradient is None:
            total_vector_gradient = vector_gradient
            detached_loss_sum = loss.detach()
        else:
            total_vector_gradient.add_(vector_gradient)
            detached_loss_sum = detached_loss_sum + loss.detach()
        observed_count += 1

    if observed_count != template_count:
        raise ValueError(
            f"expected {template_count} template losses, observed {observed_count}"
        )

    torch.autograd.backward(
        tensors=layer_vectors,
        grad_tensors=total_vector_gradient,
    )
    return float((detached_loss_sum / template_count).item())

def is_better_validation_candidate(
    current_accuracy: float,
    current_loss: float,
    best_accuracy: Optional[float],
    best_loss: Optional[float],
) -> bool:
    """Prefer validation accuracy; use validation loss only for exact ties."""
    if best_accuracy is None:
        return True
    if current_accuracy != best_accuracy:
        return current_accuracy > best_accuracy
    if best_loss is None:
        return True
    return current_loss < best_loss

def _detached_cpu_state_dict(state_dict):
    """Copy a tensor state dict to CPU so checkpoints do not retain live storage."""
    return {
        key: value.detach().cpu().clone()
        for key, value in state_dict.items()
    }

def atomic_torch_save(payload: dict, path) -> None:
    """Write a torch checkpoint atomically within the destination directory."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{target}.tmp.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
def soft_cross_supcon_loss(
    anchor_features: torch.Tensor,
    key_features: torch.Tensor,
    anchor_soft_labels: torch.Tensor,
    key_soft_labels: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Soft-label contrastive loss for mixed anchors and clean keys.

    Unlike the symmetric mixed-to-mixed loss, the matching matrix diagonal is
    meaningful here: row ``i`` is a mixed anchor while column ``i`` is its
    corresponding clean key.  It therefore remains in both the denominator and
    the soft positive target.
    """

    if temperature <= 0:
        raise ValueError("contrastive temperature must be positive")
    if anchor_features.ndim != 2 or key_features.ndim != 2:
        raise ValueError("anchor and key features must both have shape [N, D]")
    if anchor_features.shape != key_features.shape:
        raise ValueError("anchor and key features must have identical shapes")
    if anchor_soft_labels.ndim != 2 or key_soft_labels.ndim != 2:
        raise ValueError("anchor and key labels must both have shape [N, C]")
    if anchor_soft_labels.shape != key_soft_labels.shape:
        raise ValueError("anchor and key labels must have identical shapes")
    if anchor_features.shape[0] != anchor_soft_labels.shape[0]:
        raise ValueError("feature and label batch sizes do not match")

    batch_size = anchor_features.shape[0]
    if batch_size < 2:
        return anchor_features.new_tensor(0.0, requires_grad=False)

    anchors = F.normalize(anchor_features.float(), dim=-1)
    keys = F.normalize(key_features.float(), dim=-1)
    logits = anchors @ keys.T / temperature
    log_prob = F.log_softmax(logits, dim=1)

    positive_weights = (
        anchor_soft_labels.float() @ key_soft_labels.float().T
    )
    weight_sum = positive_weights.sum(dim=1)
    valid = weight_sum > 1e-8
    if not bool(valid.any()):
        return anchor_features.new_tensor(0.0, requires_grad=False)

    positive_log_prob = (
        (log_prob * positive_weights).sum(dim=1)
        / weight_sum.clamp_min(1e-8)
    )
    return -positive_log_prob[valid].mean()


def build_last_mean_optimizer(gpt2_model, projection_layer, gpt2_lr, projection_lr, weight_decay):
    groups = []
    roles = (
        ("gpt2", list(gpt2_model.named_parameters()), gpt2_lr),
        ("projection", [(n, p) for n, p in projection_layer.named_parameters()
                        if not n.startswith("down_mean.")], projection_lr),
        ("mean", list(projection_layer.down_mean.named_parameters()), projection_lr),
    )
    for name, parameters, lr in roles:
        decay, no_decay = _split_named_decay_parameters(parameters)
        for suffix, params, wd in (("decay", decay, weight_decay), ("no_decay", no_decay, 0.0)):
            if params:
                groups.append(dict(name=f"{name}_{suffix}", params=params, lr=lr, weight_decay=wd))
    return torch.optim.AdamW(groups, foreach=False)


def last_mean_config(args, gpt2_model, llama_model):
    return {
        "use_lora": True, "gpt2_model_name": args.gpt2_model_name,
        "llama_model_name": args.model_name,
        "gpt2_hidden_size": gpt2_model.config.hidden_size,
        "llama_hidden_size": llama_model.config.hidden_size,
        "llama_num_layers": llama_model.config.num_hidden_layers,
        "projection_output_size": llama_model.config.hidden_size * llama_model.config.num_hidden_layers,
        "projection_bias": True, "bottleneck_dim": None,
        "lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "lora_target_modules": list(GPT2_LORA_TARGET_MODULES),
        "readout_architecture": "joint_256", "readout_type": "joint",
        "joint_rank": 256, "last_rank": None, "mean_rank": None,
        "readout_fusion_mode": "convex", "readout_mean_alpha": args.readout_mean_alpha,
        "readout_fusion_norm": "none", "readout_fusion_norm_eps": 1e-12,
        "source_layer": "final", "source_layer_count": 1,
        "source_last_selector": "last_non_padding",
        "source_mean_selector": "masked_mean_all_non_padding",
        "mean_includes_answer_prefix": True, "activation": "identity", "imix_locus": "z_last",
        "optimizer": "adamw", "gpt2_learning_rate": args.gpt2_learning_rate,
        "projection_learning_rate": args.learning_rate, "mean_learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "weight_fv": args.weight_fv,
        "injection_position": "last_prompt_token", "injection_steps": "first_answer_token",
        "effective_batch_size": args.contrastive_batch_size,
        "microbatch_size": getattr(args, "batch_size", 1),
        "llama_batch": getattr(args, "llama_batch", False),
        "gradient_checkpointing": getattr(args, "gradient_checkpointing", False),
        "epochs": args.epochs, "seed": args.seed,
        "use_contrastive": args.use_contrastive, "use_imix": args.use_imix,
        "contrastive_lambda": args.contrastive_lambda,
        "contrastive_temperature": args.contrastive_temperature,
        "contrastive_level": args.contrastive_level,
        "contrastive_sampler": "random", "contrastive_lambda_end": args.contrastive_lambda_end,
        "imix_alpha": args.imix_alpha, "imix_key_mode": args.imix_key_mode,
        "imix_permutation_mode": "random", "memory_efficient_template_backward": True,
    }


def build_last_mean_checkpoint(gpt2_model, projection_layer, config, epoch, **metrics):
    from peft import get_peft_model_state_dict
    return {
        "checkpoint_format": CHECKPOINT_FORMAT, "epoch_number": epoch,
        "gpt2_lora_state_dict": _detached_cpu_state_dict(get_peft_model_state_dict(gpt2_model)),
        "projection_layer_state_dict": _detached_cpu_state_dict(projection_layer.state_dict()),
        "adaptive_training_config": dict(config),
        "metrics": {key: float(value) for key, value in metrics.items()},
    }


def load_last_mean_checkpoint(checkpoint, model_name, device, num_layers, hidden_size):
    from transformers import GPT2Model
    from peft import set_peft_model_state_dict
    config = checkpoint["adaptive_training_config"]
    required = {
        "readout_architecture": "joint_256", "readout_fusion_mode": "convex",
        "readout_fusion_norm": "none", "joint_rank": 256,
        "llama_num_layers": num_layers, "llama_hidden_size": hidden_size,
        "source_layer": "final", "source_last_selector": "last_non_padding",
        "source_mean_selector": "masked_mean_all_non_padding",
        "mean_includes_answer_prefix": True, "activation": "identity",
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"Incompatible checkpoint {key}: {config.get(key)!r}; expected {expected!r}")
    if tuple(config["lora_target_modules"]) != GPT2_LORA_TARGET_MODULES:
        raise ValueError("Incompatible GPT-2 LoRA target modules")
    model = GPT2Model.from_pretrained(model_name)
    if model.config.hidden_size != config["gpt2_hidden_size"]:
        raise ValueError("GPT-2 hidden size does not match checkpoint")
    model = apply_gpt2_lora(model, rank=config["lora_rank"],
                            alpha=config["lora_alpha"], dropout=config["lora_dropout"])
    projection = LastMeanProjection(model.config.hidden_size, num_layers * hidden_size,
                                    rank=config["joint_rank"], mean_alpha=config["readout_mean_alpha"])
    set_peft_model_state_dict(model, checkpoint["gpt2_lora_state_dict"])
    projection.load_state_dict(checkpoint["projection_layer_state_dict"])
    return model.to(device).eval(), projection.to(device).eval()
