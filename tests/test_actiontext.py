"""Open-vocabulary legal-action text representation."""

from typing import cast

import torch

from anvil.encoder.actiontext import ACTION_TEXT_FEATURES, ActionTextEncoder, action_text_tokens
from anvil.schemas.tensors import Example
from anvil.training.dataset import action_tokens, canonical_action_text, collate, norm_sa


def test_action_text_tokens_are_deterministic_and_open_vocabulary():
    a = action_text_tokens("{T}: Add {U}.")
    b = action_text_tokens("{T}: Add {W}.")
    assert a == action_text_tokens("{T}: Add {U}.")
    assert len(a) == ACTION_TEXT_FEATURES
    assert a != b
    assert action_text_tokens("") == [0] * ACTION_TEXT_FEATURES
    assert max(action_text_tokens("Troll of Khazad-dûm - Creature 6 / 5")) <= 32768


def test_action_text_tokenizer_golden_ids_pin_checkpoint_semantics():
    assert action_text_tokens("{T}: Add {U}.")[:12] == [
        20084,
        24016,
        30077,
        30766,
        27987,
        20084,
        21422,
        30077,
        21975,
        15962,
        11360,
        26174,
    ]
    assert action_text_tokens("Play land")[:8] == [
        16112,
        26351,
        5817,
        9871,
        27177,
        16444,
        9379,
        6390,
    ]


def test_forge_text_normalization_removes_only_incidental_variation():
    assert norm_sa("Play land (X=3)   ") == "Play land"
    assert canonical_action_text(
        "Play land by Icetill Explorer (31)"
    ) == canonical_action_text("Play land by Icetill Explorer (132)")
    assert action_tokens("Play land by Icetill Explorer (31)") == action_tokens(
        "Play land by Icetill Explorer (132)"
    )
    assert action_tokens("{T}: Add {U}.") != action_tokens("{T}: Add {W}.")


def test_long_action_features_cover_the_tail():
    prefix = "Counter target spell. " * 20
    assert action_text_tokens(prefix + "Draw a card.") != action_text_tokens(
        prefix + "Create a Treasure token."
    )


def test_legacy_action_vocab_checkpoint_has_a_clear_boundary():
    from anvil.policy.model import AnvilNet

    net = object.__new__(AnvilNet)
    try:
        AnvilNet.load_compat(net, {"sa_emb.weight": torch.zeros(2, 2)})
    except RuntimeError as error:
        assert "legacy fixed-action-vocabulary checkpoint" in str(error)
    else:
        raise AssertionError("legacy checkpoint was accepted")


def test_action_text_encoder_handles_padding_and_novel_text():
    torch.manual_seed(0)
    encoder = ActionTextEncoder(32)
    tokens = torch.tensor(
        [
            [
                action_text_tokens("Lightning Bolt deals 3 damage to any target."),
                action_text_tokens("Counter target spell unless its controller pays {1}."),
            ]
        ]
    )
    out = encoder(tokens)
    assert out.shape == (1, 2, 32)
    assert torch.isfinite(out).all()
    assert not torch.equal(out[:, 0], out[:, 1])


def test_collate_pads_candidate_text_axis():
    def ex(texts):
        n = 1
        z = lambda value: torch.tensor(value, dtype=torch.int64)  # noqa: E731
        return {
            "entities": torch.zeros(n, 18),
            "ent_emb": z([-1]),
            "globals": torch.zeros(10),
            "players": torch.zeros(2, 6),
            "history": torch.full((8, 3), -1, dtype=torch.int64),
            "cand_rows": z([-1] + [0] * len(texts)),
            "cand_text": z([action_text_tokens("")] + [action_text_tokens(t) for t in texts]),
            "label": z(0),
            "label_row": z(-1),
            "tgt_kind": torch.full((5,), -1, dtype=torch.int64),
            "tgt_idx": torch.full((5,), -1, dtype=torch.int64),
            "x_val": z(-1),
            "task": z(0),
            "bool_label": z(-1),
            "num_label": z(-1),
            "num_lo": z(0),
            "num_hi": z(17),
            "ctx_row": z(-1),
            "forced": z(0),
            "has_outcome": z(0),
            "won": z(0),
            **{k: z([]) for k in (
                "cmb_rows", "cmb_count", "cmb_count_label", "blk_atk_rows",
                "atk_label", "atk_tgt_kind", "atk_tgt_idx", "blk_label",
            )},
        }

    batch = collate(
        cast(
            list[Example],
            [
                ex(["Play land"]),
                ex(["{T}: Add {U}.", "Counter target spell unless its controller pays {1}."]),
            ],
        )
    )
    assert batch["cand_text"].shape == (2, 3, ACTION_TEXT_FEATURES)
    assert bool(batch["cand_text"][0, 1].any())
    assert not bool(batch["cand_text"][0, 2].any())
