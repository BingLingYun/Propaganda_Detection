"""End-to-end container for the two-stage neural baseline."""

from __future__ import annotations

import torch.nn as nn

from src.models.two_stage.classifier import ZenSpanTechniqueClassifier
from src.models.two_stage.proposal import SpanProposalDetector


class TwoStageSpanPipeline(nn.Module):
    """Checkpointable container for separately trained stage 1 and stage 2."""

    def __init__(
        self,
        proposal_detector: SpanProposalDetector,
        technique_classifier: ZenSpanTechniqueClassifier,
    ) -> None:
        super().__init__()
        self.proposal_detector = proposal_detector
        self.technique_classifier = technique_classifier

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        num_labels: int,
        span_hidden_size: int = 256,
        span_mlp_layers: int = 1,
        dropout: float = 0.1,
        head_architecture: str = "independent",
    ) -> "TwoStageSpanPipeline":
        proposal = SpanProposalDetector.from_pretrained(
            model_name=model_name,
            span_hidden_size=span_hidden_size,
            span_mlp_layers=span_mlp_layers,
            dropout=dropout,
        )
        classifier = ZenSpanTechniqueClassifier.from_pretrained(
            model_name=model_name,
            num_labels=num_labels,
            dropout=dropout,
            head_architecture=head_architecture,
        )
        return cls(proposal_detector=proposal, technique_classifier=classifier)
