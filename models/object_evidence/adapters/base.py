"""Adapter contract and identity alignment shared by detector integrations."""

from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch

from ..types import ObjectEvidenceBatch


class ObjectEvidenceAdapter(ABC):
    @abstractmethod
    def capture(self):
        """Return a context manager which passively captures a detector feature."""


def align_by_object_id(
    clean: ObjectEvidenceBatch, fault: ObjectEvidenceBatch
) -> Tuple[ObjectEvidenceBatch, ObjectEvidenceBatch]:
    """Inner-join two branches by (batch, object id), preserving clean order."""
    fault_positions = {}
    for index, (batch, object_id) in enumerate(zip(fault.batch_indices.tolist(), fault.object_ids)):
        key = (int(batch), object_id)
        if key in fault_positions:
            raise ValueError(f"duplicate fault object identity {key}")
        fault_positions[key] = index
    clean_indexes, fault_indexes = [], []
    seen = set()
    for index, (batch, object_id) in enumerate(zip(clean.batch_indices.tolist(), clean.object_ids)):
        key = (int(batch), object_id)
        if key in seen:
            raise ValueError(f"duplicate clean object identity {key}")
        seen.add(key)
        if key in fault_positions:
            clean_indexes.append(index)
            fault_indexes.append(fault_positions[key])
    clean_index = torch.tensor(clean_indexes, dtype=torch.long, device=clean.tokens.device)
    fault_index = torch.tensor(fault_indexes, dtype=torch.long, device=fault.tokens.device)
    return clean.select(clean_index), fault.select(fault_index)
