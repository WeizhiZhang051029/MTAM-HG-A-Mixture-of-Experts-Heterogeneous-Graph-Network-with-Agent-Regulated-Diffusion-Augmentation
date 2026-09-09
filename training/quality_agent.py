from __future__ import annotations
import torch
import torch.nn as nn


PAPER_CBTG_METRIC_NAMES = ("RMSE", "MAE", "MAPE", "ONE_MINUS_R2")

PAPER_CBTG_STATE_COMPONENTS = (
    "overall_mean",
    "run_std",
    "cluster_value",
    "cluster_variance",
)
AGENT_FEEDBACK_FEATURE_NAMES = tuple(
    f"{metric.lower()}_{component}" for metric in PAPER_CBTG_METRIC_NAMES for component in PAPER_CBTG_STATE_COMPONENTS
)


class SyntheticQualityAgent(nn.Module):
    """Score synthetic samples from process and model feedback."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        feedback_dim: int = len(AGENT_FEEDBACK_FEATURE_NAMES),
        attention_dim: int = 64,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        self.base_input_dim = int(input_dim)
        self.feedback_dim = int(feedback_dim)
        hidden_dim = int(hidden_dim)
        self.attention_dim = int(attention_dim)
        self.attention_heads = int(attention_heads)
        if self.attention_dim <= 0:
            raise ValueError("attention_dim must be positive.")
        if self.attention_heads <= 0:
            raise ValueError("attention_heads must be positive.")
        if self.attention_dim % self.attention_heads != 0:
            raise ValueError("attention_dim must be divisible by attention_heads.")
        self.sample_query = nn.Sequential(
            nn.Linear(self.base_input_dim, self.attention_dim),
            nn.LayerNorm(self.attention_dim),
            nn.GELU(),
        )
        self.feedback_value = nn.Linear(1, self.attention_dim)
        self.feedback_type_embedding = nn.Parameter(torch.zeros(self.feedback_dim, self.attention_dim))
        self.feedback_norm = nn.LayerNorm(self.attention_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=self.attention_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(float(dropout))
        head_input_dim = self.base_input_dim + self.feedback_dim + 2 * self.attention_dim
        self.net = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        y_generated: torch.Tensor,
        feedback_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        if feedback_features is None:
            feedback = x.new_zeros((batch_size, self.feedback_dim))
        else:
            feedback = feedback_features.to(device=x.device, dtype=x.dtype).reshape(batch_size, -1)
            if feedback.shape[1] != self.feedback_dim:
                raise ValueError(
                    f"Agent feedback feature dim {feedback.shape[1]} does not match expected {self.feedback_dim}."
                )
        sample_features = torch.cat(
            [x.reshape(batch_size, -1), y_generated.reshape(batch_size, -1)],
            dim=-1,
        )
        if sample_features.shape[1] != self.base_input_dim:
            raise ValueError(
                f"Agent sample feature dim {sample_features.shape[1]} does not match expected {self.base_input_dim}."
            )
        query = self.sample_query(sample_features).unsqueeze(1)
        feedback_tokens = self.feedback_value(feedback.unsqueeze(-1))
        feedback_tokens = self.feedback_norm(feedback_tokens + self.feedback_type_embedding.unsqueeze(0))
        attended, _ = self.cross_attention(query, feedback_tokens, feedback_tokens, need_weights=False)
        attended = self.attention_dropout(attended.squeeze(1))
        features = torch.cat(
            [
                sample_features,
                feedback,
                query.squeeze(1),
                attended,
            ],
            dim=-1,
        )
        return self.net(features)
