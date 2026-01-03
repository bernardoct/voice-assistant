#!/usr/bin/env python3
import json
import os
import re
import sys
from pathlib import Path
import time
from typing import Optional, Tuple, Dict, Any, List

import requests
from openai import OpenAI

from hey_george_listener import play_on_sonos, set_volume


envfile = Path.home() / ".ha_env"
if envfile.exists():
    for line in envfile.read_text().splitlines():
        if line.startswith("export "):
            k, v = line[len("export "):].split("=", 1)
            os.environ.setdefault(k, v.strip('"'))

HA_URL = os.environ.get("HA_URL", "http://192.168.1.203:8123")
HA_TOKEN = os.environ.get("HA_TOKEN")
REG_PATH = Path(os.environ.get("HA_REGISTRY_PATH", str(Path.home() / ".cache" / "ha_entities.json")))
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
OPENAI_URL = os.environ.get("OPENAI_URL", "http://$JETSON_IP:8000/v1/chat/completions")
#OPENAI_URL = os.environ.get("OPENAI_URL", "https://api.openai.com/v1/responses")

client = OpenAI(base_url=OPENAI_URL)

ALLOWED_SERVICES = {"turn_on", "turn_off"}

def norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_registry() -> Dict[str, Any]:
    return json.loads(REG_PATH.read_text())

def ha_call(domain: str, service: str, data: dict):
    if not HA_TOKEN:
        raise RuntimeError("HA_TOKEN env var not set")
    t0 = time.time()
    r = requests.post(
        f"{HA_URL}/api/services/{domain}/{service}",
        headers={"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"},
        json=data,
        timeout=10,
    )
    r.raise_for_status()
    print(f"HA call time: {time.time() - t0} s")
    return r.json()

def _entity_options(reg: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for e in reg.get("entities", []):
        friendly_name = e.get("friendly_name", "")
        if all([x not in friendly_name for x in ["Child lock", "Disable LED", "Loudness", "Crossfade", "Surround", "Night sound", "Subwoofer", "Speech enhancement"]]):
            out.append({
                "friendly_name": friendly_name,
                "supported_color_modes": e.get("supported_color_modes"),
            })
    return out

def _build_prompt(user_text: str, reg: Dict[str, Any]) -> str:
    actions = [
        "turn_on",
        "turn_off"
    ]
    entities = _entity_options(reg)
    prompt = {
        "task": "Select the best Home Assistant action and entity from the options and return JSON only. This is the "
        "transcription of a verbal command and the STT algorithm may mindunderstand words, so be mindful of words "
        "that sound similar. Still, if there's no obvious match, return empty text instead of guessing."
        "ANSWER IN ENGLISH",
        "user_text": user_text,
        "action_options": actions,
        "entity_options": entities,
        # "extra_parameter_options": EXTRA_PARAMS,
        "output_schema": {
            "service": "string, one of action_options",
            "entity_friendly_name": "string, one of entity_options.friendly_name",
            "data": "object; optional. Only include extra parameters supported by the selected entity. THERE MAY BE MULTIPLE.",
        },
    }
    return json.dumps(prompt, ensure_ascii=True)

def llm_route(user_text: str, reg: Dict[str, Any]) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY env var not set")
    prompt = _build_prompt(user_text, reg)

    t0 = time.time()
    # response = client.chat.completions.create(
    response = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
        reasoning={
            "effort": "minimal"
        }
    )
    text = response.output_text
    if not text:
        raise RuntimeError("OpenAI response missing text output")

    print(text)

    # text = _extract_text(json.loads(response.output_text))
    print(f"time to run LLM: {time.time() - t0} s")
    return json.loads(text) #_parse_json(text)

def _validate_llm_result(result: Dict[str, Any], reg: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    service = result.get("service")
    entity_friendly_name = result.get("entity_friendly_name")
    data = result.get("data") or {}

    # if service not in ALLOWED_SERVICES:
    #     raise RuntimeError(f"Invalid service from LLM: {service}")
    # if not entity_friendly_name:
    #     raise RuntimeError("Missing entity_friendly_name from LLM")
    valid_entities = {e.get("friendly_norm"): e.get("entity_id") for e in reg.get("entities", [])}
    if not entity_friendly_name:
        set_volume("media_player.living_room", 0.1)
        play_on_sonos(
            "media_player.living_room",
            "http://192.168.1.203:8123/local/capture_failed.wav",
        )
        raise RuntimeError("LLM did not return an entity_friendly_name")
    entity_id = valid_entities[entity_friendly_name.lower()]

    # Whitelist extra params
    cleaned = {}
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

def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: voice_route.py 'turn on kitchen light'", file=sys.stderr)
        return 2

    text = " ".join(sys.argv[1:])
    reg = load_registry()
    result = llm_route(text, reg)
    service, entity_id, data = _validate_llm_result(result, reg)
    # # service, entity_id, data = result['service'], result['entity_friendly_name'], result.get('data', {})
    domain = entity_id.split(".", 1)[0]
    payload = {"entity_id": entity_id}
    payload.update(data)
    ha_call(domain, service, payload)
    print(f"Executed: {domain}.{service} -> {entity_id} {data or ''}".strip())
    return 0

if __name__ == "__main__":
    sys.exit(main())
