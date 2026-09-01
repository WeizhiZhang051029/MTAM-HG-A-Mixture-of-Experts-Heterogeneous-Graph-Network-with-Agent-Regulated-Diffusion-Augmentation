from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from models.mr_lora import (
    ATTENTION_LORA_TARGETS,
    GRAPH_LORA_TARGETS,
    MR_LORA_SCOPE_FAMILIES,
    ROUTING_LORA_TARGETS,
    LoRALinear,
    inject_mr_lora,
    mr_lora_scope_families,
)
from train import build_experiment_model, configure_finetune_trainability, maybe_enable_mr_lora


class MRLoraPaperAlignmentTest(unittest.TestCase):
    def setUp(self) -> None:
        names = (
            "USE_MR_LORA",
            "MR_LORA_SCOPE",
            "MR_LORA_RANK_GRAPH",
            "MR_LORA_RANK_ROUTING",
            "MR_LORA_ALPHA_GRAPH",
            "MR_LORA_ALPHA_ROUTING",
            "MR_LORA_DROPOUT",
            "MR_LORA_TRAIN_OUTPUT_HEAD",
            "D_MODEL",
            "GRAPH_EMBED_DIM",
            "GRAPH_BACKBONE_LAYERS",
            "DROPOUT",
        )
        self._original = {name: getattr(config, name) for name in names}
        config.USE_MR_LORA = True
        config.MR_LORA_SCOPE = "graph_attention_routing"
        config.MR_LORA_RANK_GRAPH = 8
        config.MR_LORA_RANK_ROUTING = 4
        config.MR_LORA_ALPHA_GRAPH = 16.0
        config.MR_LORA_ALPHA_ROUTING = 8.0
        config.MR_LORA_DROPOUT = 0.05
        config.MR_LORA_TRAIN_OUTPUT_HEAD = False
        config.D_MODEL = 32
        config.GRAPH_EMBED_DIM = 32
        config.GRAPH_BACKBONE_LAYERS = 1
        config.DROPOUT = 0.0

    def tearDown(self) -> None:
        for name, value in self._original.items():
            setattr(config, name, value)

    @staticmethod
    def _model() -> torch.nn.Module:
        return build_experiment_model()

    def test_full_scope_uses_paper_rank_scaling_and_dropout(self) -> None:
        model = self._model()

        summary = maybe_enable_mr_lora(model)
        policy = configure_finetune_trainability(model)
        lora_modules = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, LoRALinear)
        }

        self.assertGreater(summary["graph_modules"], 0)
        self.assertGreater(summary["attention_modules"], 0)
        self.assertGreater(summary["routing_modules"], 0)
        self.assertEqual(summary["total_modules"], len(lora_modules))
        for name, module in lora_modules.items():
            with self.subTest(module=name):
                if any(name.endswith(target) for target in GRAPH_LORA_TARGETS + ATTENTION_LORA_TARGETS):
                    self.assertEqual((module.rank, module.alpha), (8, 16.0))
                elif any(name.endswith(target) for target in ROUTING_LORA_TARGETS):
                    self.assertEqual((module.rank, module.alpha), (4, 8.0))
                else:
                    self.fail(f"Unclassified LoRA target: {name}")
                self.assertIsInstance(module.dropout, torch.nn.Dropout)
                self.assertAlmostEqual(module.dropout.p, 0.05)
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        self.assertTrue(policy["mr_lora"])
        self.assertTrue(trainable)
        self.assertTrue(all(".lora_" in name for name in trainable))

        x = torch.randn(3, len(config.input_node_names(config.USE_EL_AS_INPUT)))
        model(x)["mu"].square().mean().backward()
        lora_gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if ".lora_" in name
        ]
        self.assertTrue(lora_gradients)
        self.assertTrue(all(gradient is not None for gradient in lora_gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in lora_gradients if gradient is not None))

    def test_release_exposes_only_the_full_paper_scope(self) -> None:
        self.assertEqual(tuple(MR_LORA_SCOPE_FAMILIES), ("graph_attention_routing",))
        self.assertEqual(
            mr_lora_scope_families("graph_attention_routing"),
            {"graph", "attention", "routing"},
        )
        with self.assertRaisesRegex(ValueError, "uses graph_attention_routing"):
            mr_lora_scope_families("graph_only")

    def test_partial_adapter_injection_controls_are_not_available(self) -> None:
        with self.assertRaisesRegex(TypeError, "unexpected keyword argument 'include_graph'"):
            inject_mr_lora(
                torch.nn.Linear(2, 1),
                graph_rank=8,
                routing_rank=4,
                graph_alpha=16.0,
                routing_alpha=8.0,
                include_graph=False,
            )

    def test_zero_initialized_adapters_preserve_the_pretrained_forward_pass(self) -> None:
        torch.manual_seed(123)
        model = self._model().eval()
        x = torch.randn(2, len(config.input_node_names(config.USE_EL_AS_INPUT)))
        with torch.no_grad():
            before = model(x)["mu"].clone()

        maybe_enable_mr_lora(model)
        model.eval()
        with torch.no_grad():
            after = model(x)["mu"].clone()

        self.assertTrue(torch.allclose(before, after, atol=1.0e-6, rtol=1.0e-6))


if __name__ == "__main__":
    unittest.main()
