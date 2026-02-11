"""
Reusable Plotly chart components for the FL demo UI.
"""

from typing import List, Dict, Any
import plotly.graph_objects as go
import plotly.express as px


# ── colour palette ────────────────────────────────────────────────────────────
C_BLUE   = "#1f77b4"
C_GREEN  = "#2ca02c"
C_ORANGE = "#ff7f0e"
C_RED    = "#d62728"
C_PURPLE = "#9467bd"


def create_accuracy_chart(rounds: List[int], accuracy: List[float]) -> go.Figure:
    """Line chart showing global accuracy per FL round."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rounds, y=[a * 100 for a in accuracy],
        mode="lines+markers",
        name="Accuracy",
        line=dict(color=C_GREEN, width=3),
        marker=dict(size=10, color=C_GREEN),
    ))
    fig.update_layout(
        title="Global Model Accuracy per Round",
        xaxis_title="Round",
        yaxis_title="Accuracy (%)",
        yaxis=dict(range=[0, 100]),
        template="plotly_white",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    if rounds:
        fig.add_annotation(
            x=rounds[-1], y=accuracy[-1] * 100,
            text=f"{accuracy[-1]*100:.1f}%",
            showarrow=True, arrowhead=2, bgcolor=C_GREEN, font=dict(color="white"),
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
            line=dict(color=C_ORANGE, width=3),
            marker=dict(size=8),
        ))
    if eval_loss:
        fig.add_trace(go.Scatter(
            x=rounds[:len(eval_loss)], y=eval_loss,
            mode="lines+markers", name="Evaluation Loss",
            line=dict(color=C_RED, width=3, dash="dash"),
            marker=dict(size=8),
        ))
    fig.update_layout(
        title="Loss per Round",
        xaxis_title="Round",
        yaxis_title="Loss",
        template="plotly_white",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(x=0.7, y=0.95),
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
            x=0.5, y=0.5, showarrow=False, font=dict(size=16),
        )
        fig.update_layout(height=300, template="plotly_white")
        return fig
    fig = go.Figure(go.Pie(
        labels=names, values=counts,
        marker=dict(colors=[C_BLUE, C_ORANGE]),
        textinfo="label+percent+value",
        hole=0.4,
    ))
    fig.update_layout(
        title="Patient Distribution by Hospital",
        template="plotly_white",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def create_metrics_gauge(value: float, title: str, fmt: str = "{:.1f}%") -> go.Figure:
    """Gauge chart for a single metric."""
    display = value * 100 if "%" in fmt else value
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=display,
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100] if "%" in fmt else [0, 1]},
            "bar": {"color": C_GREEN},
            "steps": [
                {"range": [0, 50], "color": "#ffcccc"},
                {"range": [50, 75], "color": "#fff0cc"},
                {"range": [75, 100], "color": "#ccffcc"},
            ],
        },
        number={"suffix": "%" if "%" in fmt else "", "valueformat": ".1f"},
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_round_progress(current: int, total: int) -> go.Figure:
    """Horizontal bar showing round progress."""
    pct = (current / total * 100) if total > 0 else 0
    fig = go.Figure(go.Bar(
        x=[pct], y=["Progress"],
        orientation="h",
        marker_color=C_BLUE,
        text=[f"Round {current}/{total} ({pct:.0f}%)"],
        textposition="inside",
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 100], title=""),
        yaxis=dict(visible=False),
        height=80,
        margin=dict(l=10, r=10, t=10, b=10),
        template="plotly_white",
    )
    return fig
