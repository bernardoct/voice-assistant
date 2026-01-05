#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

from assistant_env import load_settings
from ha_client import call_service, play_on_sonos, set_volume

ALLOWED_SERVICES = {"turn_on", "turn_off"}


def norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_registry(registry_path: str) -> Dict[str, Any]:
    return json.loads(Path(registry_path).read_text())


def ha_call(settings, domain: str, service: str, data: dict) -> None:
    t0 = time.time()
    call_service(settings, domain, service, data)
    print(f"HA call time: {time.time() - t0} s")


def _entity_options(reg: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    mode_to_param = {
        "brightness": "brightness_pct",
        "color_temp_kelvin": "color_temp_kelvin",
    }
    for entity in reg.get("entities", []):
        friendly_name = entity.get("friendly_name", "")
        if all(
            x not in friendly_name
            for x in [
                "Child lock",
                "Disable LED",
                "Loudness",
                "Crossfade",
                "Surround",
                "Night sound",
                "Subwoofer",
                "Speech enhancement",
            ]
        ):
            supported_modes = entity.get("supported_color_modes") or []
            extra_params = [
                mode_to_param[m] for m in supported_modes if m in mode_to_param
            ]
            out.append(
                {
                    "friendly_name": friendly_name,
                    "extra_parameters": extra_params,
                }
            )
    return out


def _build_prompt(user_text: str, reg: Dict[str, Any]) -> str:
    actions = ["turn_on", "turn_off"]
    entities = _entity_options(reg)
    prompt = {
        "task": "Select the best Home Assistant action and entity from the options and return JSON only. This is the "
        "transcription of a verbal command and the STT algorithm may misunderstand words, so be mindful of words "
        "that sound similar. Still, if there's no obvious match, return empty text instead of guessing. "
        "ANSWER IN ENGLISH",
        "user_text": user_text,
        "action_options": actions,
        "entity_options": entities,
        "output_schema": {
            "service": "string, one of action_options",
            "entity_friendly_name": "string, one of entity_options.friendly_name",
            "data": "object; optional. Only include keys from selected entity_options.extra_parameters. THERE MAY BE MULTIPLE.",
        },
    }
    return json.dumps(prompt, ensure_ascii=True, indent=2)


def _llm_request(llm_url: str, llm_model: str, llm_api_key: str, prompt: str) -> str:
    # 1. Check prompt
    print("Prompt to LLM:")
    print(prompt)

    # 2. Define Headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer local-anything"
    }

    # 3. Define Payload
    payload = {
        "model": llm_model,
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    # 4. Send Request
    try:
        # Using 'json=' automatically serializes the dictionary and sets content-type
        response = requests.post(llm_url, headers=headers, json=payload)
        
        # Check for HTTP errors
        response.raise_for_status()
        
        # Print the parsed JSON response
        print(json.dumps(response.json(), indent=2))

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
    response.raise_for_status()
    payload = response.json()
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("LLM response missing choices")
    content = choices[0].get("message", {}).get("content", "")
    if not content:
        raise RuntimeError("LLM response missing content")
    return content.strip()


def llm_route(settings, user_text: str, reg: Dict[str, Any]) -> Dict[str, Any]:
    prompt = _build_prompt(user_text, reg)

    t0 = time.time()
    text = _llm_request(settings.llm_url, settings.llm_model, settings.llm_api_key, prompt)
    print(text)
    print(f"time to run LLM: {time.time() - t0} s")
    return json.loads(text)


def _validate_llm_result(
    result: Dict[str, Any],
    reg: Dict[str, Any],
    settings,
) -> Tuple[str, str, Dict[str, Any]]:
    service = result.get("service")
    entity_friendly_name = result.get("entity_friendly_name")
    data = result.get("data") or {}

    if service not in ALLOWED_SERVICES:
        raise RuntimeError(f"Invalid service from LLM: {service}")
    if not entity_friendly_name:
        set_volume(settings, "media_player.living_room", 0.1)
        play_on_sonos(
            settings,
            "media_player.living_room",
            "http://192.168.1.203:8123/local/capture_failed.wav",
        )
        raise RuntimeError("LLM did not return an entity_friendly_name")

    valid_entities = {e.get("friendly_norm"): e.get("entity_id") for e in reg.get("entities", [])}
    friendly_norm = norm(entity_friendly_name)
    entity_id = valid_entities.get(friendly_norm)
    if not entity_id:
        raise RuntimeError(f"Entity not found for friendly name: {entity_friendly_name}")

    cleaned: Dict[str, Any] = {}
    if "brightness_pct" in data:
        try:
            brightness = int(data["brightness_pct"])
        except (TypeError, ValueError):
            raise RuntimeError("brightness_pct must be an integer")
        brightness = max(1, min(100, brightness))
        cleaned["brightness_pct"] = brightness
    if "color_temp_kelvin" in data:
        try:
            color_temp = int(data["color_temp_kelvin"])
        except (TypeError, ValueError):
            raise RuntimeError("color_temp_kelvin must be an integer")
        color_temp = max(2300, min(4000, color_temp))
        cleaned["color_temp_kelvin"] = color_temp
    return service, entity_id, cleaned


def handle_text(settings, text: str) -> None:
    reg = load_registry(str(settings.registry_path))
    result = llm_route(settings, text, reg)
    try:
        service, entity_id, data = _validate_llm_result(result, reg, settings)
    except RuntimeError as e:
        print(f"Validation error: {e}")
        return
    domain = entity_id.split(".", 1)[0]
    payload = {"entity_id": entity_id}
    payload.update(data)
    ha_call(settings, domain, service, payload)
    print(f"Executed: {domain}.{service} -> {entity_id} {data or ''}".strip())


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: voice_route.py 'turn on kitchen light'", file=sys.stderr)
        return 2

    settings = load_settings()
    text = " ".join(sys.argv[1:])
    handle_text(settings, text)
    return 0


if __name__ == "__main__":
    # sys.exit(main())
    main()
