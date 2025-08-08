"""
Enhanced Evaluator wrapper providing a live dashboard for real-time monitoring.
Uses DefectEvaluator as the core metrics calculator and maintains a metrics buffer
that can be visualized via Plotly Dash.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

try:
    import dash  # type: ignore
    from dash import dcc, html  # type: ignore
    import plotly.graph_objects as go  # type: ignore
except Exception:  # Dash/Plotly may be optional at runtime
    dash = None
    dcc = None
    html = None
    go = None

from .evaluator import DefectEvaluator


class EnhancedEvaluator:
    """A light wrapper adding live-dashboard capability around DefectEvaluator.

    Minimal by design: doesn't change metric computations, only collects and exposes
    a metrics buffer suitable for live plotting.
    """

    def __init__(self, evaluator: DefectEvaluator):
        self.evaluator = evaluator
        # Use a list of dicts where each entry contains: timestamp, mAP (and optionally others)
        self.metrics_buffer: List[Dict[str, Any]] = []

    def update_with_batch(self, predictions: List[Dict[str, Any]], targets: List[Dict[str, Any]], image_paths: List[str]):
        """Forward updates to the wrapped evaluator and optionally compute/record lightweight metrics.
        To keep overhead low, users can choose cadence (e.g., every N batches or per epoch).
        """
        self.evaluator.update(predictions, targets, image_paths)

    def record_snapshot(self, extra: Optional[Dict[str, Any]] = None):
        """Compute quick metrics snapshot and push to the buffer.
        Only mAP is computed here for performance; users can extend as needed.
        """
        try:
            mAP = float(self.evaluator.compute_map())
        except Exception:
            mAP = 0.0
        entry: Dict[str, Any] = {
            'timestamp': datetime.now(),
            'mAP': mAP,
        }
        if extra:
            entry.update(extra)
        self.metrics_buffer.append(entry)

    def create_live_dashboard(self):
        """Create a Plotly Dash app for real-time evaluation dashboard.
        Returns the Dash app instance. Requires `dash` and `plotly` to be installed.
        """
        if dash is None or go is None:
            raise RuntimeError("Plotly Dash is not available. Please install 'dash' and 'plotly'.")

        app = dash.Dash(__name__)

        # Build initial figure from existing buffer
        x_vals = [e['timestamp'] for e in self.metrics_buffer]
        y_vals = [e.get('mAP', 0.0) for e in self.metrics_buffer]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines+markers',
            name='mAP over time'
        ))

        # Add J&J target line
        fig.add_hline(y=0.90, line_dash="dash", annotation_text="J&J Target ≥90%")

        app.layout = html.Div([
            html.H2("Real-time Evaluation Dashboard"),
            dcc.Graph(id='map-graph', figure=fig),
        ])

        return app
