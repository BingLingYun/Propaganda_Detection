"""Vectorized multi-head BIO2 CRF with hard transition constraints."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


O_TAG = 0
B_TAG = 1
I_TAG = 2


@dataclass(frozen=True)
class CRFReduction:
    NONE: str = "none"
    SUM: str = "sum"
    MEAN: str = "mean"


def assert_bio2_legal(tags: list[int]) -> None:
    """Raise if a decoded BIO2 sequence contains START->I or O->I."""
    prev = None
    for idx, tag in enumerate(tags):
        if tag not in (O_TAG, B_TAG, I_TAG):
            raise ValueError(f"Unknown BIO tag id at position {idx}: {tag}")
        if tag == I_TAG and (prev is None or prev == O_TAG):
            raise ValueError(f"Illegal BIO2 transition at position {idx}: {prev}->{tag}")
        prev = tag


class MultiHeadConstrainedCRF(nn.Module):
    """Independent 3-state CRFs for all technique heads.

    Emissions use shape ``[batch, seq_len, num_heads, 3]`` and tags use
    ``[batch, seq_len, num_heads]``. The CRF enforces BIO2 constraints by
    adding a fixed large negative score to START->I and O->I every time the
    forward algorithm or Viterbi decoding runs.
    """

    def __init__(self, num_heads: int, forbidden_score: float = -10000.0) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        self.num_heads = num_heads
        self.num_tags = 3
        self.forbidden_score = float(forbidden_score)
        self.transitions = nn.Parameter(torch.empty(num_heads, 3, 3))
        self.start_transitions = nn.Parameter(torch.empty(num_heads, 3))
        self.end_transitions = nn.Parameter(torch.empty(num_heads, 3))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def constrained_transitions(self) -> torch.Tensor:
        constrained = self.transitions
        mask = constrained.new_zeros((1, 3, 3))
        mask[:, O_TAG, I_TAG] = self.forbidden_score
        return constrained + mask

    def constrained_start_transitions(self) -> torch.Tensor:
        constrained = self.start_transitions
        mask = constrained.new_zeros((1, 3))
        mask[:, I_TAG] = self.forbidden_score
        return constrained + mask

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
        reduction: str = CRFReduction.MEAN,
    ) -> torch.Tensor:
        """Return negative log-likelihood.

        ``mask`` can be ``[B, L]`` or ``[B, L, C]``. Masked tokens are compacted
        before CRF scoring, so special tokens at the beginning/end are safe.
        """
        emissions, tags, mask, head_ids = self._flatten_and_compact(emissions, tags, mask)
        if emissions.numel() == 0:
            empty = self.transitions.sum() * 0.0
            return empty if reduction != CRFReduction.NONE else empty.reshape(1)[:0]

        nll = self._log_partition(emissions, mask, head_ids) - self._gold_score(
            emissions, tags, mask, head_ids
        )
        if reduction == CRFReduction.NONE:
            return nll
        if reduction == CRFReduction.SUM:
            return nll.sum()
        if reduction == CRFReduction.MEAN:
            return nll.mean()
        raise ValueError(f"Unsupported reduction: {reduction}")

    @torch.no_grad()
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[list[int]]]:
        """Viterbi decode into ``tags[batch][head]`` over valid tokens only."""
        self._validate_emissions(emissions)
        batch_size, seq_len, num_heads, _ = emissions.shape
        mask = self._expand_mask(mask, batch_size, seq_len, num_heads)
        flat_emissions = emissions.permute(0, 2, 1, 3).reshape(batch_size * num_heads, seq_len, 3)
        flat_mask = mask.permute(0, 2, 1).reshape(batch_size * num_heads, seq_len)
        head_ids = torch.arange(num_heads, device=emissions.device).repeat(batch_size)

        keep = flat_mask.any(dim=1)
        result: list[list[list[int]]] = [[[] for _ in range(num_heads)] for _ in range(batch_size)]
        if not keep.any():
            return result

        compact_emissions, compact_mask, source_indices = self._compact(flat_emissions, flat_mask)
        kept_heads = head_ids[source_indices]
        starts = self.constrained_start_transitions()[kept_heads]
        ends = self.end_transitions[kept_heads]
        trans = self.constrained_transitions()[kept_heads]

        score = starts + compact_emissions[:, 0]
        history: list[torch.Tensor] = []
        for pos in range(1, compact_emissions.shape[1]):
            next_score = score[:, :, None] + trans
            best_score, best_tag = next_score.max(dim=1)
            best_score = best_score + compact_emissions[:, pos]
            active = compact_mask[:, pos].unsqueeze(1)
            score = torch.where(active, best_score, score)
            history.append(best_tag)

        score = score + ends
        best_last = score.argmax(dim=1)
        lengths = compact_mask.long().sum(dim=1)

        for row in range(compact_emissions.shape[0]):
            length = int(lengths[row].item())
            tags = [int(best_last[row].item())]
            for hist_pos in range(length - 2, -1, -1):
                previous = history[hist_pos][row, tags[-1]]
                tags.append(int(previous.item()))
            tags.reverse()
            assert_bio2_legal(tags)
            flat_index = int(source_indices[row].item())
            batch_index = flat_index // num_heads
            head_index = flat_index % num_heads
            result[batch_index][head_index] = tags
        return result

    def _gold_score(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
        head_ids: torch.Tensor,
    ) -> torch.Tensor:
        starts = self.constrained_start_transitions()[head_ids]
        ends = self.end_transitions[head_ids]
        trans = self.constrained_transitions()[head_ids]
        batch_index = torch.arange(tags.shape[0], device=tags.device)

        first_tags = tags[:, 0]
        score = starts[batch_index, first_tags] + emissions[batch_index, 0, first_tags]

        for pos in range(1, tags.shape[1]):
            prev_tags = tags[:, pos - 1]
            curr_tags = tags[:, pos]
            transition_score = trans[batch_index, prev_tags, curr_tags]
            emission_score = emissions[batch_index, pos, curr_tags]
            score = score + (transition_score + emission_score) * mask[:, pos]

        lengths = mask.long().sum(dim=1)
        last_tags = tags[batch_index, lengths - 1]
        return score + ends[batch_index, last_tags]

    def _log_partition(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
        head_ids: torch.Tensor,
    ) -> torch.Tensor:
        starts = self.constrained_start_transitions()[head_ids]
        ends = self.end_transitions[head_ids]
        trans = self.constrained_transitions()[head_ids]

        score = starts + emissions[:, 0]
        for pos in range(1, emissions.shape[1]):
            next_score = score[:, :, None] + trans + emissions[:, pos].unsqueeze(1)
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[:, pos].unsqueeze(1), next_score, score)
        return torch.logsumexp(score + ends, dim=1)

    def _flatten_and_compact(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_emissions(emissions)
        batch_size, seq_len, num_heads, _ = emissions.shape
        if tags.shape != (batch_size, seq_len, num_heads):
            raise ValueError(f"tags must have shape {(batch_size, seq_len, num_heads)}, got {tags.shape}")
        if tags.min().item() < 0 or tags.max().item() > 2:
            raise ValueError("CRF tags must be in {0, 1, 2}; mask out ignored tokens before calling")

        mask = self._expand_mask(mask, batch_size, seq_len, num_heads)
        flat_emissions = emissions.permute(0, 2, 1, 3).reshape(batch_size * num_heads, seq_len, 3)
        flat_tags = tags.permute(0, 2, 1).reshape(batch_size * num_heads, seq_len)
        flat_mask = mask.permute(0, 2, 1).reshape(batch_size * num_heads, seq_len)
        head_ids = torch.arange(num_heads, device=emissions.device).repeat(batch_size)

        compact_emissions, compact_mask, source_indices = self._compact(flat_emissions, flat_mask)
        compact_tags = self._gather_compact(flat_tags[source_indices], flat_mask[source_indices])
        return compact_emissions, compact_tags, compact_mask, head_ids[source_indices]

    def _validate_emissions(self, emissions: torch.Tensor) -> None:
        if emissions.dim() != 4:
            raise ValueError("emissions must have shape [batch, seq_len, num_heads, 3]")
        if emissions.shape[2] != self.num_heads or emissions.shape[3] != 3:
            raise ValueError(
                f"Expected num_heads={self.num_heads}, num_tags=3, got {emissions.shape[2:]}"
            )

    @staticmethod
    def _expand_mask(mask: torch.Tensor, batch_size: int, seq_len: int, num_heads: int) -> torch.Tensor:
        if mask.shape == (batch_size, seq_len):
            return mask.bool().unsqueeze(-1).expand(batch_size, seq_len, num_heads)
        if mask.shape == (batch_size, seq_len, num_heads):
            return mask.bool()
        raise ValueError(
            f"mask must have shape {(batch_size, seq_len)} or {(batch_size, seq_len, num_heads)}, "
            f"got {mask.shape}"
        )

    @staticmethod
    def _compact(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        keep = mask.any(dim=1)
        source_indices = keep.nonzero(as_tuple=False).flatten()
        x = x[source_indices]
        mask = mask[source_indices]
        lengths = mask.long().sum(dim=1)
        max_len = int(lengths.max().item())
        positions = torch.arange(max_len, device=x.device).unsqueeze(0)
        compact_mask = positions < lengths.unsqueeze(1)
        rank = mask.long().cumsum(dim=1) - 1
        target_rows = torch.arange(x.shape[0], device=x.device).unsqueeze(1).expand_as(mask)
        selected = mask.nonzero(as_tuple=True)
        compact_shape = (x.shape[0], max_len) + tuple(x.shape[2:])
        compact = x.new_zeros(compact_shape)
        compact[target_rows[selected], rank[selected]] = x[selected]
        return compact, compact_mask, source_indices

    @staticmethod
    def _gather_compact(tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.long().sum(dim=1)
        max_len = int(lengths.max().item())
        rank = mask.long().cumsum(dim=1) - 1
        rows = torch.arange(tags.shape[0], device=tags.device).unsqueeze(1).expand_as(mask)
        selected = mask.nonzero(as_tuple=True)
        compact = tags.new_zeros((tags.shape[0], max_len))
        compact[rows[selected], rank[selected]] = tags[selected]
        return compact
