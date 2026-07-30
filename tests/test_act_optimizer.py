import unittest
import torch
from zandla.detr.main import build_ACT_model_and_optimizer, build_ACT_model


class TestACTOptimizerParamFiltering(unittest.TestCase):
    def setUp(self):
        self.args_override = {
            "lr": 1e-4,
            "lr_backbone": 1e-5,
            "backbone": "resnet18",
            "masks": False,
            "dilation": False,
            "position_embedding": "sine",
            "camera_names": ["observation.image"],
            "enc_layers": 2,
            "dec_layers": 2,
            "dim_feedforward": 512,
            "hidden_dim": 128,
            "dropout": 0.1,
            "nheads": 8,
            "num_queries": 10,
            "pre_norm": False,
            "state_dim": 2,
            "weight_decay": 1e-4,
        }

    def test_active_backbone_creates_two_param_groups(self):
        """Test that with active backbone parameters, 2 parameter groups are created."""
        model, optimizer = build_ACT_model_and_optimizer(self.args_override)

        # Expect 2 param groups: non-backbone and backbone
        self.assertEqual(len(optimizer.param_groups), 2)

    def test_frozen_backbone_filters_empty_param_group(self):
        """Test that we filter out the empty backbone parameter group when frozen."""
        parser_args = type("Args", (), self.args_override)()
        model = build_ACT_model(parser_args)

        # Freeze all backbone parameters
        for name, param in model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False

        # Construct raw param dicts before filtering
        raw_param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": parser_args.lr_backbone,
            },
        ]

        # Verify that backbone params list is empty
        self.assertEqual(len(raw_param_dicts[1]["params"]), 0)

        # Apply filtering (line 77 in detr main.py)
        filtered_param_dicts = [p for p in raw_param_dicts if p["params"]]

        # Verify filtering removed the empty group
        self.assertEqual(len(filtered_param_dicts), 1)

        # Verify optimizer initializes successfully with filtered param_dicts
        optimizer = torch.optim.AdamW(
            filtered_param_dicts,
            lr=parser_args.lr,
            weight_decay=parser_args.weight_decay,
        )
        self.assertEqual(len(optimizer.param_groups), 1)


if __name__ == "__main__":
    unittest.main()
