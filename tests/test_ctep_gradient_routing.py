import torch
import torch.nn as nn

from analysis.ctep_gradient_routing import frozen_sequential_classifier


def test_frozen_classifier_is_exact_and_only_routes_to_input():
    torch.manual_seed(7)
    classifier = nn.Sequential(
        nn.Linear(5, 5), nn.LayerNorm(5), nn.ReLU(inplace=True), nn.Linear(5, 3)
    )
    query = torch.randn(2, 4, 5, requires_grad=True)
    canonical = classifier(query)
    routed = frozen_sequential_classifier(classifier, query)
    assert torch.equal(canonical, routed)
    loss = routed[..., 1].sum()
    gradients = torch.autograd.grad(
        loss, [query, *classifier.parameters()], allow_unused=True
    )
    assert gradients[0] is not None
    assert all(value is None for value in gradients[1:])

