import torch

from training.model_factory import ToyTransformerConfig, build_toy_transformer
from training.shard import build_transformer_shards, contiguous_shards, forward_sharded


def test_contiguous_shards_cover_all_layers() -> None:
    shards = contiguous_shards(num_layers=6, num_shards=3)
    assert [(spec.start_layer, spec.end_layer) for spec in shards] == [(0, 2), (2, 4), (4, 6)]


def test_sharded_forward_matches_unsharded_toy_transformer() -> None:
    config = ToyTransformerConfig(
        vocab_size=32,
        max_seq_len=8,
        d_model=16,
        num_heads=4,
        mlp_hidden_dim=32,
        num_layers=6,
    )
    model = build_toy_transformer(config, seed=17)
    input_ids = torch.tensor(
        [
            [1, 4, 7, 2, 9, 3, 5, 6],
            [6, 5, 3, 9, 2, 7, 4, 1],
        ],
        dtype=torch.long,
    )

    with torch.no_grad():
        reference_logits = model(input_ids)
        shards, specs = build_transformer_shards(model, num_shards=3)
        sharded_result = forward_sharded(shards, input_ids)

    assert [spec.shard_id for spec in specs] == [0, 1, 2]
    assert set(sharded_result.boundary_states) == {(0, 1), (1, 2)}
    torch.testing.assert_close(sharded_result.logits, reference_logits, atol=1e-6, rtol=1e-6)

