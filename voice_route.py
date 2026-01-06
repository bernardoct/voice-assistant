#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import requests


def norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_registry(registry_path: str) -> Dict[str, Any]:
    return json.loads(Path(registry_path).read_text())


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


def _room_options(reg: Dict[str, Any]) -> List[str]:
    return [a.get("name") for a in reg.get("areas", []) if a.get("name")]


def _build_prompt(user_text: str, reg: Dict[str, Any]) -> str:
    actions = ["turn_on", "turn_off"]
    entities = _entity_options(reg)
    rooms = _room_options(reg)
    prompt = {
        "task": "Select the best Home Assistant action and target from the options and return JSON only. This is the "
        "transcription of a verbal command and the STT algorithm may misunderstand words, so be mindful of words "
        "that sound similar. If the user asks to control all lights in a room, set room_name from room_options. "
        "If the request is unrelated to home control or is unclear, set intent to reply and provide response_text. "
        "ANSWER IN ENGLISH",
        "user_text": user_text,
        "action_options": actions,
        "entity_options": entities,
        "room_options": rooms,
        "output_schema": {
            "intent": "string, either 'action' or 'reply'",
            "service": "string, one of action_options; required when intent is 'action'",
            "entity_friendly_name": "string, one of entity_options.friendly_name; optional when targeting a room",
            "room_name": "string, one of room_options; use when targeting all lights in a room",
            "data": "object; optional. Only include keys from selected entity_options.extra_parameters. THERE MAY BE MULTIPLE.",
            "response_text": "string; required when intent is 'reply'",
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
        "Authorization": f"Bearer {llm_api_key or 'local-anything'}",
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
