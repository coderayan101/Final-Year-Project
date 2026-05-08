"""
Disease severity scoring based on Grad-CAM heatmap analysis.

Calculates the percentage of leaf area affected by analyzing the heatmap
from Grad-CAM, then assigns a severity level:
    - Mild     : < 10% affected area
    - Moderate : 10-40% affected area
    - Severe   : > 40% affected area

Usage:
    from src.severity import compute_severity
    severity = compute_severity(heatmap)
    # {'level': 'Moderate', 'percentage': 28.5, 'color': '#f59e0b'}
"""

import numpy as np

# Threshold above which a heatmap pixel counts as "affected"
ACTIVATION_THRESHOLD = 0.35


def compute_severity(heatmap: np.ndarray) -> dict:
    """
    Compute disease severity from a Grad-CAM heatmap.

    Args:
        heatmap: 2D numpy array with values in [0, 1], as returned by
                 gradcam.compute_heatmap().

    Returns:
        dict with keys:
            level      : str   - "Healthy", "Mild", "Moderate", or "Severe"
            percentage : float - percentage of leaf area affected (0-100)
            color      : str   - hex color for UI display
    """
    if heatmap is None or heatmap.size == 0:
        return {"level": "Unknown", "percentage": 0.0, "color": "#999999"}

    total_pixels = heatmap.size
    affected_pixels = np.sum(heatmap >= ACTIVATION_THRESHOLD)
    percentage = round(float(affected_pixels / total_pixels) * 100, 2)

    if percentage < 10:
        level, color = "Mild", "#22c55e"
    elif percentage <= 40:
        level, color = "Moderate", "#f59e0b"
    else:
        level, color = "Severe", "#ef4444"

    return {
        "level": level,
        "percentage": percentage,
        "color": color,
    }
