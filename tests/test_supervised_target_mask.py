import torch

from krasnal.supervised_target_mask import (
    IGNORE_IN_SUPERVISED_TARGET,
    LOSS_IGNORE_INDEX,
    apply_supervised_loss_mask,
)
from krasnal.tokens import (
    ELO_1500_1599_ID,
    IS_CHECK_ID,
    OPP_MATERIAL_TOKENS,
    TC_BLITZ_INC_ID,
)


def test_apply_supervised_loss_mask_matches_collate_policy():
    y = torch.tensor(
        [IS_CHECK_ID, ELO_1500_1599_ID, TC_BLITZ_INC_ID, 500],
        dtype=torch.long,
    )
    masked = apply_supervised_loss_mask(y)

    assert masked.tolist() == [
        LOSS_IGNORE_INDEX,
        LOSS_IGNORE_INDEX,
        LOSS_IGNORE_INDEX,
        500,
    ]


def test_ignore_set_includes_qa_questions():
    assert IS_CHECK_ID in IGNORE_IN_SUPERVISED_TARGET
    assert ELO_1500_1599_ID in IGNORE_IN_SUPERVISED_TARGET
    assert TC_BLITZ_INC_ID in IGNORE_IN_SUPERVISED_TARGET


def test_material_tokens_remain_supervised_targets():
    token_id = OPP_MATERIAL_TOKENS["<opp_mat_39>"]
    masked = apply_supervised_loss_mask(torch.tensor([token_id]))

    assert masked.tolist() == [token_id]
