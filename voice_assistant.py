#!/usr/bin/env python3
from __future__ import annotations

import multiprocessing as mp
import time
from typing import Any, Dict, Tuple

from assistant_env import load_settings
from ha_client import call_service, set_volume, speak_on_sonos
from hey_george_listener import run_listener
from voice_route import llm_route, load_registry, norm

ALLOWED_SERVICES = {"turn_on", "turn_off"}


def ha_call(settings, domain: str, service: str, data: dict) -> None:
    t0 = time.time()
    call_service(settings, domain, service, data)
    print(f"HA call time: {time.time() - t0} s")


def _clean_data(data: Dict[str, Any]) -> Dict[str, Any]:
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
    return cleaned


def _find_room(reg: Dict[str, Any], room_name: str) -> Dict[str, Any] | None:
    room_norm = norm(room_name)
    for area in reg.get("areas", []):
        if norm(str(area.get("name", ""))) == room_norm:
            return area
    return None


def _validate_llm_result(
    result: Dict[str, Any],
    reg: Dict[str, Any],
) -> Tuple[str, str, Dict[str, Any]]:
    service = result.get("service")
    entity_friendly_name = result.get("entity_friendly_name")
    room_name = result.get("room_name")
    data = result.get("data") or {}

    if service not in ALLOWED_SERVICES:
        raise RuntimeError(f"Invalid service from LLM: {service}")

    if room_name:
        area = _find_room(reg, room_name)
        if not area:
            raise RuntimeError(f"Room not found for name: {room_name}")
        area_id = area.get("area_id")
        payload = {}
        if area_id:
            payload["area_id"] = area_id
        else:
            light_entities = area.get("light_entities") or []
            if not light_entities:
                raise RuntimeError(f"No lights found for room: {room_name}")
            payload["entity_id"] = light_entities
        payload.update(_clean_data(data))
        return "light", service, payload

    if not entity_friendly_name:
        raise RuntimeError("LLM did not return an entity_friendly_name")

    valid_entities = {e.get("friendly_norm"): e.get("entity_id") for e in reg.get("entities", [])}
    friendly_norm = norm(entity_friendly_name)
    entity_id = valid_entities.get(friendly_norm)
    if not entity_id:
        raise RuntimeError(f"Entity not found for friendly name: {entity_friendly_name}")

    payload = {"entity_id": entity_id}
    payload.update(_clean_data(data))
    domain = entity_id.split(".", 1)[0]
    return domain, service, payload


def _speak_reply(settings, message: str) -> None:
    message = message.strip()
    if not message:
        return
    set_volume(settings, "media_player.living_room", 0.1)
    speak_on_sonos(settings, "media_player.living_room", message)


def handle_text(settings, text: str) -> None:
    reg = load_registry(str(settings.registry_path))
    result = llm_route(settings, text, reg)
    response_text = (result.get("response_text") or "").strip()
    intent = result.get("intent")
    if intent == "reply" or (not result.get("service") and response_text):
        _speak_reply(settings, response_text or "I am not sure how to help with that.")
        return
    try:
        domain, service, payload = _validate_llm_result(result, reg)
    except RuntimeError as e:
        print(f"Validation error: {e}")
        if response_text:
            _speak_reply(settings, response_text)
        else:
            _speak_reply(settings, "Sorry, I could not figure out that request.")
        return
    ha_call(settings, domain, service, payload)
    print(f"Executed: {domain}.{service} -> {payload}".strip())


def router_process(text_queue: mp.Queue[str]) -> None:
    settings = load_settings()
    while True:
        text = text_queue.get()
        if text is None:
            break
        handle_text(settings, text)


def start_router(
    ctx: mp.context.BaseContext,
    queue: mp.Queue[str] | None = None,
) -> tuple[mp.Process, mp.Queue[str]]:
    if queue is None:
        queue = ctx.Queue()
    proc = ctx.Process(target=router_process, args=(queue,), daemon=True)
    proc.start()
    return proc, queue


def main() -> None:
    settings = load_settings()
    ctx = mp.get_context("spawn")
    router_proc, router_queue = start_router(ctx)

    def on_text(text: str) -> None:
        nonlocal router_proc, router_queue
        if not router_proc.is_alive():
            print("Router process died; restarting.")
            router_proc, router_queue = start_router(ctx, router_queue)
        router_queue.put(text)

    run_listener(settings, on_text)


if __name__ == "__main__":
    main()
