import pytest
import torch

from vyrex_ai.models import VyrexConfig, VyrexTransformer


@pytest.fixture
def small_model():
    config = VyrexConfig(
        vocab_size=128,
        hidden_size=48,
        num_layers=2,
        num_heads=6,
        intermediate_size=96,
        max_seq_len=16,
        dropout=0.0,
        tie_embeddings=True,
    )
    return VyrexTransformer(config)


def test_output_shape(small_model):
    tokens = torch.randint(0, 128, (2, 8))
    logits, loss = small_model(tokens)

    assert logits.shape == (2, 8, 128)
    assert loss is None


def test_causal_attention(small_model):
    small_model.eval()

    original = torch.randint(0, 128, (1, 8))
    modified = original.clone()
    modified[:, 4:] = torch.randint(0, 128, (1, 4))

    with torch.no_grad():
        original_logits, _ = small_model(original)
        modified_logits, _ = small_model(modified)

    assert torch.allclose(
        original_logits[:, :4],
        modified_logits[:, :4],
        atol=1e-6,
    ), "Future tokens influenced earlier predictions"


def test_next_token_loss(small_model):
    tokens = torch.randint(0, 128, (2, 8))
    targets = torch.randint(0, 128, (2, 8))

    logits, loss = small_model(tokens, targets)

    assert logits.shape == (2, 8, 128)
    assert loss is not None
    assert torch.isfinite(loss)


def test_gradient_flow(small_model):
    tokens = torch.randint(0, 128, (2, 8))
    targets = torch.randint(0, 128, (2, 8))

    _, loss = small_model(tokens, targets)
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in small_model.parameters()
        if parameter.requires_grad
    ]

    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_tied_embeddings(small_model):
    assert (
        small_model.token_embeddings.weight
        is small_model.lm_head.weight
    )


def test_rejects_oversized_sequence(small_model):
    tokens = torch.randint(0, 128, (1, 17))

    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        small_model(tokens)


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        VyrexConfig(hidden_size=50, num_heads=6)


def test_model_has_trainable_parameters(small_model):
    count = sum(
        parameter.numel()
        for parameter in small_model.parameters()
        if parameter.requires_grad
    )

    assert count > 0
    print("SMALL MODEL PARAMETERS:", count)

@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"num_heads": 0}, ValueError, "must be positive"),
        ({"num_layers": 0}, ValueError, "must be positive"),
        ({"vocab_size": 0}, ValueError, "must be positive"),
        ({"intermediate_size": 0}, ValueError, "must be positive"),
        ({"max_seq_len": 0}, ValueError, "must be positive"),
        ({"hidden_size": 50, "num_heads": 6}, ValueError, "divisible"),
        ({"hidden_size": 48, "num_heads": 16}, ValueError, "even for RoPE"),
        ({"dropout": float("nan")}, ValueError, "finite"),
        ({"dropout": float("inf")}, ValueError, "finite"),
        ({"dropout": 1.0}, ValueError, "finite"),
        ({"tie_embeddings": 1}, TypeError, "boolean"),
        ({"num_heads": 2.5}, TypeError, "integer"),
        ({"hidden_size": True}, TypeError, "integer"),
    ],
)
def test_invalid_configuration_values(kwargs, error, message):
    with pytest.raises(error, match=message):
        VyrexConfig(**kwargs)


def test_default_configuration():
    config = VyrexConfig()

    assert config.vocab_size == 16000
    assert config.hidden_size == 384
    assert config.num_layers == 6
    assert config.num_heads == 6
    assert config.intermediate_size == 1024
    assert config.max_seq_len == 1024
    assert config.tie_embeddings is True
