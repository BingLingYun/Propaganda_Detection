"""Stage 1 BILOU token tagger for HALT-PROP style span proposals."""

from __future__ import annotations

import torch
import torch.nn as nn


O_TAG = 0
B_TAG = 1
I_TAG = 2
L_TAG = 3
U_TAG = 4
IGNORE = -100
BILOU_TAGS = ["O", "B", "I", "L", "U"]


class SpanProposalDetector(nn.Module):
    """Binary propaganda span proposal detector trained as BILOU tagging."""

    def __init__(
        self,
        encoder: nn.Module,
        dropout: float = 0.1,
        num_tags: int = 5,
    ) -> None:
        super().__init__()
        if num_tags != 5:
            raise ValueError("The BILOU proposal detector expects exactly 5 tags.")
        self.encoder = encoder
        self.num_tags = num_tags
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder.config.hidden_size, num_tags)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        span_hidden_size: int = 256,
        span_mlp_layers: int = 1,
        dropout: float = 0.1,
    ) -> "SpanProposalDetector":
        del span_hidden_size, span_mlp_layers
        from transformers import AutoModel

        encoder = AutoModel.from_pretrained(model_name)
        return cls(encoder=encoder, dropout=dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        class_weight: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.classifier(self.dropout(hidden))
        output = {"logits": logits}
        if labels is not None:
            if labels.shape != logits.shape[:2]:
                raise ValueError(f"labels must have shape {logits.shape[:2]}, got {labels.shape}")
            labels_for_loss = labels
            if valid_mask is not None:
                labels_for_loss = labels.masked_fill(~valid_mask.bool(), IGNORE)
            loss_fn = nn.CrossEntropyLoss(weight=class_weight, ignore_index=IGNORE)
            output["loss"] = loss_fn(logits.view(-1, self.num_tags), labels_for_loss.view(-1))
        return output
