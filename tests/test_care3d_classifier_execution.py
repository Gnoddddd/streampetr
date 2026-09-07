import copy

import pytest
import torch
from torch import nn

from models.care3d_classifier_execution import (
    CLASSIFIER_EXECUTION_POLICY,
    DEPLOYED_QUERY_COUNT,
    install_deployment_shape_forward,
)


class RecordingClassifier(nn.Module):
    def __init__(self, input_dim=8, output_dim=4):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.shapes = []

    def forward(self, value):
        self.shapes.append(tuple(value.shape))
        return self.linear(value)


def test_two_dimensional_calls_use_deployed_900_query_shape_and_match_reference():
    torch.manual_seed(7)
    classifier = RecordingClassifier()
    reference = copy.deepcopy(classifier)
    install_deployment_shape_forward(classifier)

    query = torch.randn(11, 8)
    output = classifier(query)

    packed = torch.cat(
        (query, query.new_zeros((DEPLOYED_QUERY_COUNT - len(query), query.shape[-1]))),
        dim=0,
    ).unsqueeze(0)
    expected = reference(packed)[0, : len(query)]

    assert classifier.shapes[-1] == (1, DEPLOYED_QUERY_COUNT, 8)
    assert torch.equal(output, expected)
    assert classifier._care3d_classifier_execution_policy == CLASSIFIER_EXECUTION_POLICY


def test_packed_execution_preserves_input_gradients_for_router_training():
    classifier = RecordingClassifier()
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    install_deployment_shape_forward(classifier)

    query = torch.randn(13, 8, requires_grad=True)
    classifier(query).square().mean().backward()

    assert query.grad is not None
    assert torch.isfinite(query.grad).all()
    assert float(query.grad.abs().sum()) > 0.0


def test_native_three_dimensional_detector_calls_pass_through_unchanged():
    torch.manual_seed(11)
    classifier = RecordingClassifier()
    reference = copy.deepcopy(classifier)
    install_deployment_shape_forward(classifier)

    query = torch.randn(1, DEPLOYED_QUERY_COUNT, 8)
    output = classifier(query)
    expected = reference(query)

    assert classifier.shapes[-1] == (1, DEPLOYED_QUERY_COUNT, 8)
    assert torch.equal(output, expected)


def test_packed_execution_rejects_batch_larger_than_deployed_query_count():
    classifier = RecordingClassifier()
    install_deployment_shape_forward(classifier)

    with pytest.raises(ValueError, match="exceeds query_count"):
        classifier(torch.randn(DEPLOYED_QUERY_COUNT + 1, 8))
