"""
Visualization Module
All Plotly-based charts for the Streamlit dashboard.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Any

SENTIMENT_COLORS = {
    "Positive": "#2ecc71",
    "Negative": "#e74c3c",
    "Neutral":  "#f39c12",
}

BG_COLOR  = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(255,255,255,0.08)"
TEXT_COLOR = "#e0e0e0"
FONT_FAMILY = "Inter, Segoe UI, sans-serif"

LAYOUT_BASE = dict(
    paper_bgcolor=BG_COLOR,
    plot_bgcolor=BG_COLOR,
    font=dict(color=TEXT_COLOR, family=FONT_FAMILY),
    margin=dict(l=20, r=20, t=40, b=20),
)


# ─── Sentiment Distribution ──────────────────────────────────────────────────

def plot_sentiment_donut(counts: Dict[str, int]) -> go.Figure:
    """Donut chart of sentiment distribution."""
    labels = list(counts.keys())
    values = list(counts.values())
    colors = [SENTIMENT_COLORS.get(l, "#aaa") for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#1a1a2e", width=3)),
        textinfo="percent+label",
        textfont=dict(size=14),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Sentiment Distribution", font=dict(size=18), x=0.5),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        annotations=[dict(text=f"<b>{sum(values)}</b><br>total", x=0.5, y=0.5,
                          font_size=16, showarrow=False, font_color=TEXT_COLOR)],
    )
    return fig


def plot_sentiment_bar(counts: Dict[str, int]) -> go.Figure:
    """Animated horizontal bar chart of sentiment counts."""
    labels = list(counts.keys())
    values = list(counts.values())
    colors = [SENTIMENT_COLORS.get(l, "#aaa") for l in labels]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)")),
        text=[f"  {v}" for v in values],
        textposition="outside",
        hovertemplate="<b>%{y}</b>: %{x} reviews<extra></extra>",
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Review Counts by Sentiment", font=dict(size=16), x=0.5),
        xaxis=dict(showgrid=True, gridcolor=GRID_COLOR, title="Count"),
        yaxis=dict(showgrid=False),
        bargap=0.3,
    )
    return fig


# ─── Confusion Matrix ─────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, labels: List[str]) -> go.Figure:
    """
    Annotated heatmap confusion matrix with sample counts, total test set share,
    and class-specific recall to provide complete context alongside overall accuracy.
    """
    total_samples = int(cm.sum())
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    annotations = []
    customdata = np.empty((len(labels), len(labels), 4), dtype=object)

    for i, row in enumerate(cm):
        actual_total = int(row.sum())
        for j, val in enumerate(row):
            val_int = int(val)
            pct_total = (val_int / total_samples * 100) if total_samples > 0 else 0
            recall_pct = (val_int / actual_total * 100) if actual_total > 0 else 0
            customdata[i, j] = [val_int, pct_total, recall_pct, actual_total]

            if i == j:
                cell_text = f"<b>{val_int}</b><br><span style='font-size:10px; color:#bbf7d0;'>({recall_pct:.0f}% class recall)</span>"
            else:
                if val_int > 0:
                    cell_text = f"<b>{val_int}</b><br><span style='font-size:10px; color:#fca5a5;'>({recall_pct:.0f}% error)</span>"
                else:
                    cell_text = "<b style='color:#666688;'>0</b>"

            annotations.append(dict(
                x=labels[j], y=labels[i],
                text=cell_text,
                showarrow=False,
                font=dict(color="white", size=12),
            ))

    fig = go.Figure(go.Heatmap(
        z=cm_norm,
        x=labels, y=labels,
        customdata=customdata,
        colorscale=[[0, "#12122b"], [0.4, "#4338ca"], [0.75, "#6c5ce7"], [1, "#10b981"]],
        showscale=True,
        colorbar=dict(
            title=dict(text="Class Recall", side="right", font=dict(size=11, color=TEXT_COLOR)),
            tickformat=".0%",
            tickfont=dict(color=TEXT_COLOR, size=10),
        ),
        hovertemplate=(
            "<b>Actual:</b> %{y}<br>"
            "<b>Predicted:</b> %{x}<br>"
            "<b>Count:</b> %{customdata[0]} reviews<br>"
            "<b>Share of Test Set:</b> %{customdata[1]:.1f}% (" + str(total_samples) + " total)<br>"
            "<b>Class Recall (%{y}):</b> %{customdata[2]:.1f}% (%{customdata[0]}/%{customdata[3]})<extra></extra>"
        ),
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=f"Confusion Matrix (N = {total_samples} test reviews)", font=dict(size=16), x=0.5),
        xaxis=dict(title="Predicted Sentiment", tickfont=dict(size=13)),
        yaxis=dict(title="Actual Sentiment", tickfont=dict(size=13), autorange="reversed"),
        annotations=annotations,
        height=400,
    )
    return fig


# ─── Metrics Radar ────────────────────────────────────────────────────────────

def plot_metrics_radar(metrics: Dict[str, float]) -> go.Figure:
    """Radar/spider chart of model evaluation metrics."""
    keys = ["Accuracy", "Precision", "Recall", "F1-Score"]
    vals = [
        metrics.get("accuracy", 0) * 100,
        metrics.get("precision", 0) * 100,
        metrics.get("recall", 0) * 100,
        metrics.get("f1_score", 0) * 100,
    ]
    vals_closed = vals + [vals[0]]
    keys_closed = keys + [keys[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_closed, theta=keys_closed, fill="toself",
        fillcolor="rgba(108,92,231,0.2)",
        line=dict(color="#6c5ce7", width=2),
        name="Performance",
        hovertemplate="%{theta}: %{r:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        polar=dict(
            bgcolor=BG_COLOR,
            radialaxis=dict(visible=True, range=[0, 100], gridcolor=GRID_COLOR,
                            tickcolor=TEXT_COLOR, tickfont=dict(color=TEXT_COLOR)),
            angularaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TEXT_COLOR, size=13)),
        ),
        title=dict(text="Model Performance Metrics", font=dict(size=16), x=0.5),
        showlegend=False,
    )
    return fig


# ─── Model Comparison ────────────────────────────────────────────────────────

def plot_model_comparison(comp_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing all models."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    palette = ["#6c5ce7", "#00cec9", "#fd79a8", "#fdcb6e"]

    fig = go.Figure()
    for i, model in enumerate(comp_df["Model"].tolist()):
        row = comp_df[comp_df["Model"] == model].iloc[0]
        vals = [row.get(m, 0) for m in metrics]
        fig.add_trace(go.Bar(
            name=model.split(" (")[0],
            x=metrics,
            y=vals,
            marker=dict(color=palette[i % len(palette)]),
            text=[f"{v:.1f}%" for v in vals],
            textposition="outside",
        ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Model Comparison", font=dict(size=16), x=0.5),
        barmode="group",
        yaxis=dict(range=[0, 115], title="Score (%)", gridcolor=GRID_COLOR),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
        bargap=0.15, bargroupgap=0.05,
    )
    return fig


# ─── Top Features ─────────────────────────────────────────────────────────────

def plot_top_features(features_dict: Dict[str, List]) -> go.Figure:
    """Horizontal bar chart of top TF-IDF features per sentiment class."""
    classes = list(features_dict.keys())
    colors = [SENTIMENT_COLORS.get(c, "#aaa") for c in classes]
    n_cols = len(classes)

    fig = make_subplots(rows=1, cols=n_cols,
                        subplot_titles=[f"Top Words: {c}" for c in classes],
                        shared_xaxes=False)

    for col_idx, (cls, feats) in enumerate(features_dict.items(), start=1):
        top_feats = feats[:15]
        words = [f[0] for f in reversed(top_feats)]
        scores = [f[1] for f in reversed(top_feats)]
        color = SENTIMENT_COLORS.get(cls, "#aaa")

        fig.add_trace(go.Bar(
            x=scores, y=words, orientation="h",
            marker=dict(color=color, opacity=0.85),
            showlegend=False,
            name=cls,
            hovertemplate=f"<b>%{{y}}</b><br>Weight: %{{x:.4f}}<extra>{cls}</extra>",
        ), row=1, col=col_idx)

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Most Predictive Words per Sentiment", font=dict(size=16), x=0.5),
        height=400,
    )
    for i in range(1, n_cols + 1):
        fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, row=1, col=i)
        fig.update_yaxes(showgrid=False, row=1, col=i)

    return fig


# ─── Probability Gauge ────────────────────────────────────────────────────────

def plot_probability_gauge(proba: Dict[str, float], prediction: str) -> go.Figure:
    """Three gauges showing confidence per class."""
    classes_order = ["Positive", "Negative", "Neutral"]
    classes = [c for c in classes_order if c in proba] + [c for c in proba if c not in classes_order]

    fig = make_subplots(
        rows=1, cols=len(classes),
        specs=[[{"type": "indicator"}] * len(classes)],
        subplot_titles=classes,
    )
    for i, cls in enumerate(classes, start=1):
        val = proba.get(cls, 0) * 100
        color = SENTIMENT_COLORS.get(cls, "#aaa")
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=val,
            number=dict(suffix="%", font=dict(color=color, size=22)),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor=TEXT_COLOR),
                bar=dict(color=color),
                bgcolor="rgba(255,255,255,0.05)",
                borderwidth=0,
                steps=[dict(range=[0, 50], color="rgba(255,255,255,0.03)"),
                       dict(range=[50, 100], color="rgba(255,255,255,0.06)")],
            ),
        ), row=1, col=i)

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Prediction Confidence", font=dict(size=16), x=0.5),
        height=240,
    )
    return fig


# ─── Sentiment Timeline ───────────────────────────────────────────────────────

def plot_sentiment_timeline(df: pd.DataFrame, label_col: str = "sentiment") -> go.Figure:
    """Rolling sentiment counts over the dataset index (simulates timeline)."""
    df = df.copy().reset_index(drop=True)
    window = max(10, len(df) // 20)
    sentiments = ["Positive", "Negative", "Neutral"]
    colors = [SENTIMENT_COLORS[s] for s in sentiments]

    fig = go.Figure()
    for s, c in zip(sentiments, colors):
        rolling = (df[label_col] == s).rolling(window, min_periods=1).mean() * 100
        fig.add_trace(go.Scatter(
            x=df.index, y=rolling, name=s,
            line=dict(color=c, width=2),
            fill="tozeroy" if s == "Positive" else None,
            fillcolor=f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.07)",
            hovertemplate=f"{s}: %{{y:.1f}}%<extra></extra>",
        ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=f"Sentiment Trend (rolling avg, window={window})", font=dict(size=16), x=0.5),
        xaxis=dict(title="Review Index", gridcolor=GRID_COLOR),
        yaxis=dict(title="Sentiment %", gridcolor=GRID_COLOR, range=[0, 100]),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
        hovermode="x unified",
    )
    return fig
