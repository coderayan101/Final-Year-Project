"""
Automated Telegram alerts for critical disease detection.

Sends a Telegram message with disease name, severity, confidence,
treatment advice, and Grad-CAM image when severity is Moderate or Severe.

Setup:
    1. Open Telegram, search @BotFather, send /newbot, follow prompts
    2. Copy the bot token
    3. Send a message to your bot, then visit:
       https://api.telegram.org/bot<TOKEN>/getUpdates
       to find your chat_id
    4. Set environment variables or edit the defaults below:
       TELEGRAM_BOT_TOKEN=your_token_here
       TELEGRAM_CHAT_ID=your_chat_id_here

Usage:
    from src.telegram_alert import send_alert
    send_alert(prediction_result, gradcam_path)
"""

import os
from pathlib import Path

import requests

# Configure these via environment variables or hardcode for demo
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

TELEGRAM_API = "https://api.telegram.org/bot{token}"


def _is_configured() -> bool:
    return bool(BOT_TOKEN) and bool(CHAT_ID)


def send_alert(prediction: dict, gradcam_path: Path = None) -> bool:
    """
    Send a Telegram alert with prediction details.

    Args:
        prediction: dict from MultimodalPredictor.predict() with keys:
                    disease, confidence, severity, advice, sensor_used
        gradcam_path: optional Path to the Grad-CAM overlay PNG

    Returns:
        True if message was sent successfully, False otherwise.
    """
    if not _is_configured():
        print("[telegram] Not configured - set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        return False

    severity = prediction.get("severity", {})
    disease = prediction.get("disease", "Unknown").replace("___", " - ").replace("_", " ")
    confidence = prediction.get("confidence", 0) * 100
    level = severity.get("level", "Unknown")
    percentage = severity.get("percentage", 0)
    sensor = prediction.get("sensor_used", {})
    advice = prediction.get("advice", {})

    # Build alert message
    emoji = "🔴" if level == "Severe" else "🟡"
    message = (
        f"{emoji} *CROP DISEASE ALERT* {emoji}\n"
        f"\n"
        f"*Disease:* {disease}\n"
        f"*Confidence:* {confidence:.1f}%\n"
        f"*Severity:* {level} ({percentage:.1f}% leaf area affected)\n"
        f"\n"
        f"📡 *Sensor Readings:*\n"
        f"  Temperature: {sensor.get('temperature', 0):.1f}°C\n"
        f"  Humidity: {sensor.get('humidity', 0):.1f}%\n"
        f"  Soil Moisture: {sensor.get('soil_moisture', 0):.1f}%\n"
        f"\n"
        f"💊 *Treatment:*\n"
        f"{advice.get('treatment', 'No treatment info available')}\n"
        f"\n"
        f"🛡️ *Prevention:*\n"
        f"{advice.get('prevention', 'No prevention info available')}"
    )

    api_base = TELEGRAM_API.format(token=BOT_TOKEN)

    try:
        # Send Grad-CAM image with caption if available
        if gradcam_path and Path(gradcam_path).exists():
            with open(gradcam_path, "rb") as photo:
                resp = requests.post(
                    f"{api_base}/sendPhoto",
                    data={"chat_id": CHAT_ID, "caption": message, "parse_mode": "Markdown"},
                    files={"photo": photo},
                    timeout=15,
                )
        else:
            # Text-only fallback
            resp = requests.post(
                f"{api_base}/sendMessage",
                json={"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"},
                timeout=15,
            )

        if resp.status_code == 200:
            print(f"[telegram] Alert sent: {disease} ({level})")
            return True
        else:
            print(f"[telegram] API error {resp.status_code}: {resp.text}")
            return False

    except Exception as e:
        print(f"[telegram] Failed to send: {e}")
        return False
