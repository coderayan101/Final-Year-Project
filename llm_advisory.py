"""
LLM-powered dynamic treatment advisory using Claude API.

Instead of static disease_dict lookups, sends the prediction context
(disease, severity, sensor readings) to Claude and gets personalized,
condition-aware treatment recommendations.

Setup:
    pip install anthropic
    Set environment variable: ANTHROPIC_API_KEY=your_key_here

Usage:
    from src.llm_advisory import get_llm_advice
    advice = get_llm_advice(prediction_result)
"""

import os

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


def _is_configured() -> bool:
    return bool(ANTHROPIC_API_KEY)


def get_llm_advice(prediction: dict) -> dict | None:
    """
    Get personalized treatment advice from Claude based on prediction context.

    Args:
        prediction: dict with keys disease, confidence, severity, sensor_used, advice

    Returns:
        dict with 'llm_advice' string, or None if not configured / failed.
    """
    if not _is_configured():
        return None

    try:
        import anthropic
    except ImportError:
        print("[llm] anthropic package not installed. Run: pip install anthropic")
        return None

    disease = prediction.get("disease", "Unknown").replace("___", " - ").replace("_", " ")
    confidence = prediction.get("confidence", 0) * 100
    severity = prediction.get("severity", {})
    level = severity.get("level", "Unknown")
    percentage = severity.get("percentage", 0)
    sensor = prediction.get("sensor_used", {})
    static_advice = prediction.get("advice", {})

    prompt = (
        f"You are an expert agricultural plant pathologist. A crop disease detection "
        f"system has identified the following:\n\n"
        f"Disease: {disease}\n"
        f"Confidence: {confidence:.1f}%\n"
        f"Severity: {level} ({percentage:.1f}% of leaf area affected)\n"
        f"Current Environmental Conditions:\n"
        f"  - Temperature: {sensor.get('temperature', 'N/A')}°C\n"
        f"  - Humidity: {sensor.get('humidity', 'N/A')}%\n"
        f"  - Soil Moisture: {sensor.get('soil_moisture', 'N/A')}%\n\n"
        f"Based on this diagnosis and the CURRENT environmental conditions, provide:\n"
        f"1. Immediate actions the farmer should take (specific to the severity level)\n"
        f"2. Recommended fungicide/pesticide with dosage\n"
        f"3. Cultural practices to prevent spread\n"
        f"4. How the current weather conditions affect disease progression\n"
        f"5. When to follow up or re-inspect\n\n"
        f"Keep the response practical, concise (under 250 words), and suitable for "
        f"a farmer with basic agricultural knowledge. Use simple language."
    )

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        llm_text = message.content[0].text
        return {"llm_advice": llm_text}

    except Exception as e:
        print(f"[llm] Claude API error: {e}")
        return None
