"""XLM-R multi-head BIO tagger backed by the custom constrained CRF."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.multihead_crf import MultiHeadConstrainedCRF


class XLMRMultiHeadBIOCRF(nn.Module):
    """Shared encoder plus one vectorized BIO head per propaganda technique."""

    def __init__(
        self,
        encoder: nn.Module,
        num_labels: int,
        dropout: float = 0.1,
        head_weighting: str = "none",
        negative_head_weight: float = 1.0,
        aux_bio_ce_weight: float = 0.0,
        bio_positive_weight: float = 1.0,
        forbidden_score: float = -10000.0,
    ) -> None:
        super().__init__()
        if num_labels <= 0:
            raise ValueError("num_labels must be positive")
        if head_weighting not in {"none", "inverse_sqrt"}:
            raise ValueError("head_weighting must be 'none' or 'inverse_sqrt'")
        if negative_head_weight <= 0:
            raise ValueError("negative_head_weight must be positive")
        if aux_bio_ce_weight < 0:
            raise ValueError("aux_bio_ce_weight must be non-negative")
        if bio_positive_weight <= 0:
            raise ValueError("bio_positive_weight must be positive")
        self.encoder = encoder
        self.num_labels = num_labels
        self.num_heads = num_labels
        self.head_weighting = head_weighting
        self.negative_head_weight = float(negative_head_weight)
        self.aux_bio_ce_weight = float(aux_bio_ce_weight)
        self.bio_positive_weight = float(bio_positive_weight)
        hidden_size = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels * 3)
        self.crf = MultiHeadConstrainedCRF(num_heads=num_labels, forbidden_score=forbidden_score)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> "XLMRMultiHeadBIOCRF":
        from transformers import AutoModel

        encoder = AutoModel.from_pretrained(model_name)
        return cls(encoder=encoder, num_labels=num_labels, dropout=dropout, **kwargs)

    def emissions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.classifier(self.dropout(hidden))
        return logits.view(input_ids.shape[0], input_ids.shape[1], self.num_labels, 3)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        head_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute emissions and optional CRF loss.

        ``labels`` must be ``[B, L, C]`` with values 0/1/2 at valid positions.
        ``valid_mask`` is ``[B, L]`` or ``[B, L, C]`` and excludes special tokens.
        """
        emissions = self.emissions(input_ids, attention_mask)
        output = {"emissions": emissions}
        if labels is None:
            return output
        mask = attention_mask.bool() if valid_mask is None else valid_mask.bool()
        # CRF forward/Viterbi dynamic programming is numerically sensitive. Keep
        # it in fp32 even when the encoder is under autocast/fp16.
        crf_emissions = emissions.float()
        if head_weights is None and self.negative_head_weight == 1.0:
            loss = self.crf(crf_emissions, labels.long(), mask, reduction="mean")
        else:
            losses = self.crf(crf_emissions, labels.long(), mask, reduction="none")
            flat_weights = self._sequence_loss_weights(labels, mask, head_weights, input_ids.device)
            loss = (losses * flat_weights[: losses.numel()]).sum() / flat_weights[: losses.numel()].sum().clamp_min(1e-8)
        if self.aux_bio_ce_weight > 0:
            aux_loss = self._auxiliary_bio_ce_loss(crf_emissions, labels.long(), mask, head_weights)
            loss = loss + self.aux_bio_ce_weight * aux_loss
            output["aux_bio_ce_loss"] = aux_loss
        output["loss"] = loss
        return output

    @torch.no_grad()
    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> list[list[list[int]]]:
        mask = attention_mask.bool() if valid_mask is None else valid_mask.bool()
        return self.crf.decode(self.emissions(input_ids, attention_mask).float(), mask)

    def parameter_groups(self, encoder_lr: float, task_lr: float, weight_decay: float) -> list[dict[str, Any]]:
        no_decay = ("bias", "LayerNorm.weight", "layer_norm")
        encoder_params = list(self.encoder.named_parameters())
        task_params = [(name, param) for name, param in self.named_parameters() if not name.startswith("encoder.")]

        def split(params: list[tuple[str, nn.Parameter]], lr: float) -> list[dict[str, Any]]:
            decay = [param for name, param in params if not any(key in name for key in no_decay)]
            nodecay = [param for name, param in params if any(key in name for key in no_decay)]
            return [
                {"params": decay, "lr": lr, "weight_decay": weight_decay},
                {"params": nodecay, "lr": lr, "weight_decay": 0.0},
            ]

        return split(encoder_params, encoder_lr) + split(task_params, task_lr)

    def _expand_head_weights(self, head_weights: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        if head_weights.shape != (self.num_labels,):
            raise ValueError(f"head_weights must have shape [{self.num_labels}]")
        return head_weights.to(device).repeat(batch_size)

    def _sequence_loss_weights(
        self,
        labels: torch.Tensor,
        mask: torch.Tensor,
        head_weights: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        batch_size, seq_len, num_heads = labels.shape
        if mask.shape == (batch_size, seq_len):
            expanded_mask = mask.bool().unsqueeze(-1).expand(batch_size, seq_len, num_heads)
        elif mask.shape == (batch_size, seq_len, num_heads):
            expanded_mask = mask.bool()
        else:
            raise ValueError(
                f"mask must have shape {(batch_size, seq_len)} or {(batch_size, seq_len, num_heads)}, got {mask.shape}"
            )
        flat_mask = expanded_mask.permute(0, 2, 1).reshape(batch_size * num_heads, seq_len)
        flat_positive = ((labels != 0) & expanded_mask).permute(0, 2, 1).reshape(
            batch_size * num_heads, seq_len
        ).any(dim=1)
        keep = flat_mask.any(dim=1)
        weights = flat_positive.to(dtype=torch.float32, device=device)
        weights = weights + (1.0 - weights) * self.negative_head_weight
        if head_weights is not None:
            weights = weights * self._expand_head_weights(head_weights, batch_size, device)
        return weights[keep]

    def _expanded_mask(self, mask: torch.Tensor, batch_size: int, seq_len: int, num_heads: int) -> torch.Tensor:
        if mask.shape == (batch_size, seq_len):
            return mask.bool().unsqueeze(-1).expand(batch_size, seq_len, num_heads)
        if mask.shape == (batch_size, seq_len, num_heads):
            return mask.bool()
        raise ValueError(
            f"mask must have shape {(batch_size, seq_len)} or {(batch_size, seq_len, num_heads)}, got {mask.shape}"
        )

    def _auxiliary_bio_ce_loss(
        self,
        emissions: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        head_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, seq_len, num_heads, _ = emissions.shape
        expanded_mask = self._expanded_mask(mask, batch_size, seq_len, num_heads)
        flat_valid = expanded_mask.reshape(-1)
        if not flat_valid.any():
            return emissions.sum() * 0.0

        flat_logits = emissions.reshape(-1, 3)[flat_valid]
        flat_labels = labels.reshape(-1)[flat_valid]
        losses = F.cross_entropy(flat_logits, flat_labels, reduction="none")

        weights = torch.ones_like(losses)
        weights = torch.where(flat_labels == 0, weights, weights * self.bio_positive_weight)
        if head_weights is not None:
            token_head_weights = head_weights.to(emissions.device).view(1, 1, num_heads).expand(
                batch_size, seq_len, num_heads
            )
            weights = weights * token_head_weights.reshape(-1)[flat_valid]
        return (losses * weights).sum() / weights.sum().clamp_min(1e-8)
