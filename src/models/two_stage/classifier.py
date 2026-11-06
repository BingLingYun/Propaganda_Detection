"""ZenPropaganda-style selected-span technique classifier."""

from __future__ import annotations

import torch
import torch.nn as nn


class ZenSpanTechniqueClassifier(nn.Module):
    """Classify selected spans from span text, context text, and span-token features."""

    def __init__(
        self,
        encoder: nn.Module,
        num_labels: int,
        dropout: float = 0.1,
        hidden_size: int | None = None,
        head_architecture: str = "independent",
    ) -> None:
        super().__init__()
        if num_labels <= 0:
            raise ValueError("num_labels must be positive")
        if head_architecture not in {"shared", "independent"}:
            raise ValueError("head_architecture must be 'shared' or 'independent'")
        self.encoder = encoder
        self.num_labels = num_labels
        self.head_architecture = head_architecture
        encoder_hidden = encoder.config.hidden_size
        classifier_hidden = hidden_size or encoder_hidden
        self.dropout = nn.Dropout(dropout)
        input_hidden = encoder_hidden * 2 + 1
        if head_architecture == "shared":
            self.classifier = nn.Sequential(
                nn.Linear(input_hidden, classifier_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden, num_labels),
            )
        else:
            self.classifier = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(input_hidden, classifier_hidden),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(classifier_hidden, 1),
                    )
                    for _ in range(num_labels)
                ]
            )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
        hidden_size: int | None = None,
        head_architecture: str = "independent",
    ) -> "ZenSpanTechniqueClassifier":
        from transformers import AutoModel

        encoder = AutoModel.from_pretrained(model_name)
        return cls(
            encoder=encoder,
            num_labels=num_labels,
            dropout=dropout,
            hidden_size=hidden_size,
            head_architecture=head_architecture,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_token_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        pos_weight: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls_repr = hidden[:, 0]
        span_repr = self._span_mean_pool(hidden, span_token_mask)
        span_width = span_token_mask.sum(dim=1, keepdim=True).to(hidden.dtype)
        span_width = torch.log1p(span_width) / 8.0
        features = self.dropout(torch.cat([cls_repr, span_repr, span_width], dim=-1))
        if self.head_architecture == "shared":
            logits = self.classifier(features)
        else:
            logits = torch.cat([head(features) for head in self.classifier], dim=-1)
        output = {"logits": logits}
        if labels is not None:
            if labels.shape != logits.shape:
                raise ValueError(f"labels must have shape {logits.shape}, got {labels.shape}")
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            output["loss"] = loss_fn(logits, labels.float())
        return output

    @staticmethod
    def _span_mean_pool(hidden: torch.Tensor, span_token_mask: torch.Tensor) -> torch.Tensor:
        if span_token_mask.shape[:2] != hidden.shape[:2]:
            raise ValueError(f"span_token_mask must have shape {hidden.shape[:2]}, got {span_token_mask.shape}")
        if not torch.all(span_token_mask.any(dim=1)):
            raise ValueError("each candidate must retain at least one selected-span token")
        weights = span_token_mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
