from typing import Annotated, Any

import torch
from pydantic import PlainValidator


def validate_tensor(v: Any):
    if isinstance(v, torch.Tensor):
        return v
    if isinstance(v, (list, tuple)):
        return torch.tensor(v)

    raise ValueError(f"Expected a tensor, but got {type(v)}")


AnnotatedTensor = Annotated[torch.Tensor, PlainValidator(validate_tensor)]
