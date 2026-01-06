from __future__ import annotations

from typing import Any, Dict

import requests

from assistant_env import Settings


def _require_token(settings: Settings) -> str:
    if not settings.ha_token:
        raise RuntimeError("HA_TOKEN env var not set")
    return settings.ha_token


def call_service(settings: Settings, domain: str, service: str, data: Dict[str, Any]) -> Any:
    token = _require_token(settings)
    response = requests.post(
        f"{settings.ha_url}/api/services/{domain}/{service}",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=data,
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def play_on_sonos(settings: Settings, entity_id: str, url: str) -> None:
    call_service(
        settings,
        "media_player",
        "play_media",
        {
            "entity_id": entity_id,
            "media_content_id": url,
            "media_content_type": "music",
        },
    )


def set_volume(settings: Settings, entity_id: str, level: float) -> None:
    call_service(
        settings,
        "media_player",
        "volume_set",
        {"entity_id": entity_id, "volume_level": level},
    )


def speak_on_sonos(settings: Settings, entity_id: str, message: str) -> None:
    data = {settings.tts_target_field: entity_id, "message": message}
    call_service(settings, settings.tts_domain, settings.tts_service, data)
