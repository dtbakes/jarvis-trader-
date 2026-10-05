import sys
import os
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_message(text, markdown=True):
    """
    Send a message to Derrick's Telegram. If Markdown parsing fails (an
    unpaired _ in EUR_USD or an error string is enough), retry as plain text
    so alerts are never silently dropped.
    """
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}
    if markdown:
        payload["parse_mode"] = "Markdown"
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.json().get("ok", False):
            return True
        if markdown:
            return send_message(text, markdown=False)
        print(f"[telegram error] {r.text}")
        return False
    except Exception as e:
        print(f"[telegram error] {e}")
        return False


if __name__ == "__main__":
    ok = send_message("🤖 *JARVIS online.* Notification system connected.")
    print("Sent!" if ok else "Failed to send.")