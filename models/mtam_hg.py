"""MTAM-HG with four IPOHGN experts and reliability-aware Top-2 routing."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from models.ipohgn import IPOHGNExpert

FOUR_EXPERT_NAMES = [
    "composition_property",
    "thermal_process",
    "process_coupling",
    "quality_sensitive",
]

MECHANISM_EXPERT_NODE_GROUPS = {
    "composition_property": ["C", "Mn", "S", "P", "ATh", "AWd", "HT", "FRT", "CT"],
    "thermal_process": ["JPF_PT", "HF_T", "SF_T", "SC_T", "FC1_T", "OA_T", "FC2_T", "Q_T", "HT", "FRT", "CT"],
    "process_coupling": ["RF", "BF", "CRR", "ATh", "AWd", "FS", "CT", "Q_T"],
    "quality_sensitive": ["C", "Mn", "S", "P", "ATh", "AWd", "CRR", "CT", "Q_T", "FRT"],
}


def _cfg(name: str, default):
    lower = name.lower()
    if hasattr(config, name):
        return getattr(config, name)
    return getattr(config, lower, default)


def _moe_aux_lambda() -> float:
    return float(_cfg("MOE_AUX_LAMBDA", getattr(config, "LAMBDA_MOE", 0.01)))


def router_load_balance_loss(
    gate_weights: torch.Tensor,
    gate_probs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Balance sparse expert usage while keeping soft gate probabilities exploratory."""
    num_experts = gate_weights.shape[-1]
    uniform = gate_weights.new_full((num_experts,), 1.0 / num_experts)

    importance = gate_weights.sum(dim=0)
    importance = importance / (importance.sum() + 1.0e-8)
    importance_loss = ((importance - uniform) ** 2).mean()

    load = gate_weights.mean(dim=0)
    load_loss = ((load - uniform) ** 2).mean()
    sparse_balance = importance_loss + load_loss

    prob_balance = gate_weights.new_tensor(0.0)
    entropy_reg = gate_weights.new_tensor(0.0)
    if gate_probs is not None:
        prob_importance = gate_probs.mean(dim=0)
        prob_balance = ((prob_importance - uniform) ** 2).mean()
        entropy = -(gate_probs * torch.log(gate_probs + 1.0e-8)).sum(dim=-1).mean()
        max_entropy = torch.log(gate_probs.new_tensor(float(num_experts)))
        entropy_reg = (max_entropy - entropy) / (max_entropy + 1.0e-8)

    return (
        float(_cfg("MOE_BALANCE_USAGE_LAMBDA", 1.0)) * sparse_balance
        + float(_cfg("MOE_BALANCE_PROB_LAMBDA", 0.5)) * prob_balance
        + float(_cfg("MOE_ENTROPY_REG_LAMBDA", 0.01)) * entropy_reg
    )


def _topk_sparse_weights(
    logits: torch.Tensor,
    top_k: int,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build reference-style Top-K sparse weights from gate logits."""
    scaled_logits = logits / max(float(temperature), 1.0e-6)
    topk_values, topk_indices = torch.topk(scaled_logits, k=top_k, dim=-1)
    topk_weights = torch.softmax(topk_values, dim=-1)
    sparse = torch.zeros_like(scaled_logits)
    sparse.scatter_(1, topk_indices, topk_weights)
    return sparse, topk_indices, topk_values


class ReliabilityAwareRouter(nn.Module):
    """Route samples across IPOHGN experts."""

    def __init__(
        self,
        global_dim: int,
        num_experts: int,
        top_k: int,
        process_input_dim: int | None = None,
        out_dim: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_process_features: bool = True,
        use_expert_preds: bool = True,
        use_uncertainty: bool = True,
        output_sample_confidence: bool = True,
        reason_dim: int = 4,
        reliability_routing_lambda: float = 1.0,
    ) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError("ReliabilityAwareRouter requires at least one expert.")
        if top_k < 1 or top_k > num_experts:
            raise ValueError(f"Top-K must be in [1, {num_experts}], got {top_k}.")
        self.global_dim = global_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.process_input_dim = int(process_input_dim or 0)
        self.out_dim = out_dim
        self.use_process_features = bool(use_process_features and self.process_input_dim > 0)
        self.use_expert_preds = use_expert_preds
        self.use_uncertainty = use_uncertainty
        self.output_sample_confidence = output_sample_confidence
        self.reason_dim = int(reason_dim)
        self.reliability_routing_lambda = float(reliability_routing_lambda)
        self.temperature = float(_cfg("MOE_GATE_TEMPERATURE", 1.0))

        state_dim = global_dim
        if self.use_process_features:
            state_dim += self.process_input_dim
        if use_expert_preds:
            state_dim += num_experts * out_dim
        if use_uncertainty:
            state_dim += out_dim
        self.state_dim = state_dim

        self.agent_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.routing_head = nn.Linear(hidden_dim, num_experts)
        self.confidence_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.synthetic_keep_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.training_weight_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.expert_reliability_head = nn.Sequential(nn.Linear(hidden_dim, num_experts), nn.Sigmoid())
        self.uncertainty_reason_head = nn.Sequential(nn.Linear(hidden_dim, self.reason_dim), nn.Sigmoid())

    def _build_state(
        self,
        global_hidden: torch.Tensor,
        expert_preds: torch.Tensor,
        process_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pieces = [global_hidden]
        if self.use_process_features:
            if process_features is None:
                raise ValueError("The router requires process features, but none were provided.")
            pieces.append(process_features.reshape(process_features.shape[0], -1))
        if self.use_expert_preds:
            pieces.append(expert_preds.reshape(expert_preds.shape[0], -1))
        uncertainty = expert_preds.var(dim=1, unbiased=False).reshape(expert_preds.shape[0], -1)
        if self.use_uncertainty:
            pieces.append(uncertainty)
        return torch.cat(pieces, dim=-1), uncertainty

    def forward(
        self,
        global_hidden: torch.Tensor,
        expert_preds: torch.Tensor,
        process_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        agent_state, uncertainty = self._build_state(global_hidden, expert_preds, process_features)
        encoded = self.agent_encoder(agent_state)
        raw_gate_logits = self.routing_head(encoded)
        expert_reliability = self.expert_reliability_head(encoded).clamp(1.0e-6, 1.0)
        gate_logits = raw_gate_logits + self.reliability_routing_lambda * torch.log(expert_reliability)
        scaled_logits = gate_logits / max(self.temperature, 1.0e-6)
        gate_probs = torch.softmax(scaled_logits, dim=-1)
        sparse, topk_indices, topk_values = _topk_sparse_weights(gate_logits, self.top_k, self.temperature)
        expert_weights = sparse + (gate_probs - gate_probs.detach())
        if self.output_sample_confidence:
            sample_confidence = self.confidence_head(encoded)
        else:
            sample_confidence = torch.ones(
                global_hidden.shape[0],
                1,
                device=global_hidden.device,
                dtype=global_hidden.dtype,
            )
        synthetic_keep_score = self.synthetic_keep_head(encoded)
        training_weight = self.training_weight_head(encoded)
        uncertainty_reason_vector = self.uncertainty_reason_head(encoded)
        entropy = -(gate_probs * torch.log(gate_probs + 1.0e-8)).sum(dim=-1)
        return {
            "gate_logits": gate_logits,
            "raw_gate_logits": raw_gate_logits,
            "gate_probs": gate_probs,
            "expert_weights": expert_weights,
            "topk_indices": topk_indices,
            "topk_values": topk_values,
            "sample_confidence": sample_confidence,
            "synthetic_keep_score": synthetic_keep_score,
            "training_weight": training_weight,
            "expert_reliability": expert_reliability,
            "uncertainty_reason_vector": uncertainty_reason_vector,
            "agent_state": agent_state,
            "expert_uncertainty": uncertainty,
            "agent_gate_entropy": entropy,
        }


class MTAMHG(nn.Module):
    """The paper model: four IPOHGN experts with reliability-aware Top-2 routing."""

    def __init__(
        self,
        input_dim: int,
        d_model: int | None = None,
        num_layers: int | None = None,
        use_laplace: bool = True,
    ) -> None:
        super().__init__()
        d_model = d_model or config.D_MODEL
        self.input_dim = input_dim
        self.d_model = d_model
        self.use_laplace = use_laplace
        if int(_cfg("NUM_EXPERTS", 4)) != len(FOUR_EXPERT_NAMES):
            raise ValueError("The released MTAM-HG model uses exactly four IPOHGN experts.")
        self.expert_names = list(FOUR_EXPERT_NAMES)
        self.num_experts = len(self.expert_names)
        self.top_k = int(_cfg("TOP_K", 2))
        if self.top_k != 2:
            raise ValueError("The released MTAM-HG model uses Top-2 routing.")

        self.experts = nn.ModuleList(
            [
                IPOHGNExpert(
                    input_dim=input_dim,
                    d_model=d_model,
                    num_layers=num_layers,
                    use_laplace=use_laplace,
                    enable_relation_scaling=True,
                    expert_name=name,
                    mechanism_focus_nodes=MECHANISM_EXPERT_NODE_GROUPS.get(name, config.input_node_names(config.USE_EL_AS_INPUT)),
                )
                for name in self.expert_names
            ]
        )
        self.register_buffer("mechanism_masks", self._build_mechanism_masks())
        self.gate_state_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.router = ReliabilityAwareRouter(
            global_dim=d_model,
            num_experts=self.num_experts,
            top_k=self.top_k,
            process_input_dim=input_dim,
            out_dim=1,
            hidden_dim=int(_cfg("AGENT_HIDDEN_DIM", 128)),
            dropout=float(_cfg("AGENT_DROPOUT", 0.1)),
            use_process_features=bool(_cfg("AGENT_USE_PROCESS_FEATURES", True)),
            use_expert_preds=bool(_cfg("AGENT_USE_EXPERT_PREDS", True)),
            use_uncertainty=bool(_cfg("AGENT_USE_UNCERTAINTY", True)),
            output_sample_confidence=bool(_cfg("AGENT_OUTPUT_SAMPLE_CONFIDENCE", True)),
            reason_dim=int(_cfg("AGENT_REASON_DIM", 4)),
            reliability_routing_lambda=float(_cfg("AGENT_RELIABILITY_ROUTING_LAMBDA", 1.0)),
        )
        self.aux_lambda = _moe_aux_lambda()
        self.node_names = self.experts[0].node_names
        self.input_node_names = self.experts[0].input_node_names

    def _build_mechanism_masks(self) -> torch.Tensor:
        node_names = config.active_node_names(config.USE_EL_AS_INPUT)
        input_node_names = config.input_node_names(config.USE_EL_AS_INPUT)
        node_to_idx = {name: idx for idx, name in enumerate(node_names)}
        masks = torch.zeros(self.num_experts, len(node_names), dtype=torch.float32)
        for expert_idx, name in enumerate(self.expert_names):
            selected = MECHANISM_EXPERT_NODE_GROUPS.get(name, input_node_names)
            for node in selected:
                if node in node_to_idx:
                    masks[expert_idx, node_to_idx[node]] = 1.0
            if masks[expert_idx].sum() == 0:
                masks[expert_idx, :] = 1.0
        return masks

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor] | dict[str, object]]:
        expert_outputs = [expert(x) for expert in self.experts]
        expert_preds = torch.stack([out["mu"] for out in expert_outputs], dim=1)
        expert_hidden = torch.stack([out["hidden"] for out in expert_outputs], dim=1)
        expert_A_kg = torch.stack([out["A_kg"] for out in expert_outputs], dim=0)
        expert_A0 = torch.stack([out["A0"] for out in expert_outputs], dim=0)
        hidden_norm = F.normalize(expert_hidden, dim=-1)
        similarity = torch.matmul(hidden_norm, hidden_norm.transpose(1, 2))
        eye = torch.eye(self.num_experts, device=x.device, dtype=torch.bool).unsqueeze(0)
        off_diag_similarity = similarity.masked_select(~eye)
        diversity_loss = (
            off_diag_similarity.pow(2).mean()
            if off_diag_similarity.numel()
            else similarity.new_tensor(0.0)
        )

        gate_state = self.gate_state_proj(expert_hidden.mean(dim=1))
        gate_out = self.router(gate_state, expert_preds, process_features=x)
        expert_weights = gate_out["expert_weights"].to(dtype=expert_preds.dtype)
        y_pred = (expert_preds * expert_weights.unsqueeze(-1)).sum(dim=1)

        aux_loss = router_load_balance_loss(expert_weights, gate_out["gate_probs"])
        expert_weight_forward = expert_weights.detach()
        debug: dict[str, object] = {
            "model": "mtam_hg",
            "expert_names": self.expert_names,
            "mechanism_node_groups": MECHANISM_EXPERT_NODE_GROUPS,
            "top_k": self.top_k,
            "num_experts": self.num_experts,
            "aux_lambda": self.aux_lambda,
            "expert_usage": expert_weight_forward.mean(dim=0).detach(),
            "expert_selected_rate": (expert_weight_forward > 0).float().mean(dim=0).detach(),
            "expert_weight_mean": expert_weight_forward.mean(dim=0).detach(),
        }
        if "sample_confidence" in gate_out:
            debug["sample_confidence_mean"] = gate_out["sample_confidence"].detach().mean()
        if "synthetic_keep_score" in gate_out:
            debug["synthetic_keep_score_mean"] = gate_out["synthetic_keep_score"].detach().mean()
        if "training_weight" in gate_out:
            debug["training_weight_mean"] = gate_out["training_weight"].detach().mean()
        if "expert_reliability" in gate_out:
            debug["expert_reliability_mean"] = gate_out["expert_reliability"].detach().mean()
        if "expert_uncertainty" in gate_out:
            debug["expert_uncertainty_mean"] = gate_out["expert_uncertainty"].detach().mean()
        if "agent_gate_entropy" in gate_out:
            debug["agent_gate_entropy"] = gate_out["agent_gate_entropy"].detach().mean()
        outputs: dict[str, torch.Tensor | list[torch.Tensor] | dict[str, object]] = {
            "mu": y_pred,
            "y_pred": y_pred,
            "expert_preds": expert_preds,
            "expert_weights": expert_weights,
            "gate_probs": gate_out["gate_probs"],
            "gate_logits": gate_out["gate_logits"],
            "topk_indices": gate_out["topk_indices"],
            "aux_loss": aux_loss,
            "diversity_loss": diversity_loss,
            "gate_weights": [expert_weights],
            "hidden": gate_state,
            "A_kg": expert_outputs[0]["A_kg"],
            "A_kg_experts": expert_A_kg,
            "A_het": expert_outputs[0]["A_het"],
            "A_hat": expert_outputs[0]["A_hat"],
            "A0": expert_outputs[0]["A0"],
            "A0_experts": expert_A0,
            "mechanism_masks": self.mechanism_masks,
            "expert_focus_masks": torch.stack(
                [out["mechanism_focus_mask"] for out in expert_outputs],
                dim=0,
            ),
            "debug": debug,
        }
        if "process_order_ids" in expert_outputs[0]:
            outputs["process_order_ids"] = expert_outputs[0]["process_order_ids"]
        for optional_key in (
            "sample_confidence",
            "synthetic_keep_score",
            "training_weight",
            "expert_reliability",
            "uncertainty_reason_vector",
            "expert_uncertainty",
            "agent_gate_entropy",
            "agent_state",
            "raw_gate_logits",
        ):
            if optional_key in gate_out:
                outputs[optional_key] = gate_out[optional_key]

        if self.use_laplace and all("b" in out for out in expert_outputs):
            expert_b = torch.stack([out["b"] for out in expert_outputs], dim=1)
            outputs["expert_b"] = expert_b
            outputs["b"] = (expert_b * expert_weights.unsqueeze(-1)).sum(dim=1).clamp_min(config.LAPLACE_EPS)
        return outputs
