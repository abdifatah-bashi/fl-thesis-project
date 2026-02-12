"""
Reusable Plotly chart components for the FL demo UI.
"""

from typing import List, Dict, Any
import plotly.graph_objects as go
import plotly.express as px


# ── colour palette ────────────────────────────────────────────────────────────
C_INDIGO = "#6366f1"
C_VIOLET = "#8b5cf6"
C_GREEN  = "#10b981"
C_AMBER  = "#f59e0b"
C_ROSE   = "#f43f5e"
C_SLATE  = "#475569"

_LAYOUT_COMMON = dict(
    template="plotly_white",
    font=dict(family="Inter, -apple-system, sans-serif", color="#475569"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=50, b=40),
)


def create_accuracy_chart(rounds: List[int], accuracy: List[float]) -> go.Figure:
    """Line chart showing global accuracy per FL round."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rounds, y=[a * 100 for a in accuracy],
        mode="lines+markers",
        name="Accuracy",
        line=dict(color=C_INDIGO, width=3, shape="spline"),
        marker=dict(size=9, color=C_INDIGO, line=dict(width=2, color="#fff")),
        fill="tozeroy",
        fillcolor="rgba(99,102,241,.06)",
    ))
    fig.update_layout(
        title=dict(text="Global Accuracy", font=dict(size=14, color="#1e293b")),
        xaxis_title="Round",
        yaxis_title="Accuracy (%)",
        yaxis=dict(range=[0, 100], gridcolor="#f1f3f9"),
        xaxis=dict(gridcolor="#f1f3f9"),
        height=340,
        **_LAYOUT_COMMON,
    )
    if rounds:
        fig.add_annotation(
            x=rounds[-1], y=accuracy[-1] * 100,
            text=f"<b>{accuracy[-1]*100:.1f}%</b>",
            showarrow=True, arrowhead=0, arrowcolor=C_INDIGO,
            bgcolor=C_INDIGO, font=dict(color="white", size=12),
            bordercolor=C_INDIGO, borderpad=4, borderwidth=0,
        )
    return fig


def create_loss_chart(
    rounds: List[int],
    train_loss: List[float],
    eval_loss: List[float],
) -> go.Figure:
    """Multi-line chart showing training and evaluation loss per round."""
    fig = go.Figure()
    if train_loss:
        fig.add_trace(go.Scatter(
            x=rounds[:len(train_loss)], y=train_loss,
            mode="lines+markers", name="Training Loss",
            line=dict(color=C_AMBER, width=3, shape="spline"),
            marker=dict(size=7, line=dict(width=2, color="#fff")),
        ))
    if eval_loss:
        fig.add_trace(go.Scatter(
            x=rounds[:len(eval_loss)], y=eval_loss,
            mode="lines+markers", name="Evaluation Loss",
            line=dict(color=C_ROSE, width=3, dash="dot", shape="spline"),
            marker=dict(size=7, line=dict(width=2, color="#fff")),
        ))
    fig.update_layout(
        title=dict(text="Loss per Round", font=dict(size=14, color="#1e293b")),
        xaxis_title="Round",
        yaxis_title="Loss",
        yaxis=dict(gridcolor="#f1f3f9"),
        xaxis=dict(gridcolor="#f1f3f9"),
        height=340,
        legend=dict(
            x=0.65, y=0.95,
            bgcolor="rgba(255,255,255,.8)",
            bordercolor="#e8ecf4", borderwidth=1,
            font=dict(size=11),
        ),
        **_LAYOUT_COMMON,
    )
    return fig


def create_hospital_distribution_chart(hospitals: Dict[str, Any]) -> go.Figure:
    """Pie chart showing patient distribution across hospitals."""
    names, counts = [], []
    for name, info in hospitals.items():
        if info.get("registered") and info.get("num_patients", 0) > 0:
            names.append(name.capitalize())
            counts.append(info["num_patients"])
    if not names:
        fig = go.Figure()
        fig.add_annotation(
            text="No hospitals registered yet",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#94a3b8"),
        )
        fig.update_layout(height=280, **_LAYOUT_COMMON)
        return fig
    fig = go.Figure(go.Pie(
        labels=names, values=counts,
        marker=dict(
            colors=[C_INDIGO, C_VIOLET],
            line=dict(color="#fff", width=3),
        ),
        textinfo="label+percent+value",
        textfont=dict(size=12),
        hole=0.45,
    ))
    fig.update_layout(
        title=dict(text="Patient Distribution", font=dict(size=14, color="#1e293b")),
        height=280,
        **_LAYOUT_COMMON,
    )
    return fig


def create_metrics_gauge(value: float, title: str, fmt: str = "{:.1f}%") -> go.Figure:
    """Gauge chart for a single metric."""
    display = value * 100 if "%" in fmt else value
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=display,
        title={"text": title, "font": {"size": 13, "color": "#475569"}},
        gauge={
            "axis": {"range": [0, 100] if "%" in fmt else [0, 1], "tickcolor": "#e8ecf4"},
            "bar": {"color": C_INDIGO},
            "bgcolor": "#f1f3f9",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50],  "color": "#fef2f2"},
                {"range": [50, 75], "color": "#fffbeb"},
                {"range": [75, 100], "color": "#ecfdf5"},
            ],
        },
        number={"suffix": "%" if "%" in fmt else "", "valueformat": ".1f",
                "font": {"size": 28, "color": "#1e293b", "family": "Inter"}},
    ))
    fig.update_layout(height=220, **_LAYOUT_COMMON)
    return fig


def create_round_progress(current: int, total: int) -> go.Figure:
    """Horizontal bar showing round progress."""
    pct = (current / total * 100) if total > 0 else 0
    fig = go.Figure(go.Bar(
        x=[pct], y=["Progress"],
        orientation="h",
        marker=dict(
            color=C_INDIGO,
            line=dict(width=0),
        ),
        text=[f"Round {current}/{total} ({pct:.0f}%)"],
        textposition="inside",
        textfont=dict(color="white", size=12, family="Inter"),
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 100], title="", showgrid=False),
        yaxis=dict(visible=False),
        height=64,
        **_LAYOUT_COMMON,
        margin=dict(l=10, r=10, t=6, b=6),
    )
    return fig
