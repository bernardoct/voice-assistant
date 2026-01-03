#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

from assistant_env import load_settings


def norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ha_get(ha_url: str, ha_token: str, path: str, timeout: int = 10) -> Any:
    if not ha_token:
        raise RuntimeError("HA_TOKEN env var not set")
    response = requests.get(
        f"{ha_url}{path}",
        headers={"Authorization": f"Bearer {ha_token}"},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def main() -> int:
    settings = load_settings()
    out_path = settings.registry_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    states = ha_get(settings.ha_url, settings.ha_token, "/api/states")

    entities: List[Dict[str, Any]] = []
    for st in states:
        entity_id = st.get("entity_id", "")
        if not (entity_id.startswith("light.") or entity_id.startswith("switch.")):
            continue

        attrs = st.get("attributes", {}) or {}
        friendly = attrs.get("friendly_name") or entity_id

        entities.append(
            {
                "entity_id": entity_id,
                "domain": entity_id.split(".", 1)[0],
                "friendly_name": friendly,
                "friendly_norm": norm(str(friendly)),
                "entity_norm": norm(entity_id),
                "device_class": attrs.get("device_class"),
                "supported_color_modes": attrs.get("supported_color_modes"),
            }
        )
        if "brightness" in attrs:
            supported = entities[-1]["supported_color_modes"]
            if not supported:
                entities[-1]["supported_color_modes"] = ["brightness"]
            elif "brightness" not in supported:
                supported.append("brightness")
        if "color_temp" in attrs.get("supported_color_modes", []):
            scm = entities[-1]["supported_color_modes"]
            scm.remove("color_temp")
            scm.append("color_temp_kelvin")

    by_friendly: Dict[str, List[str]] = {}
    by_entity: Dict[str, str] = {}
    for e in entities:
        by_entity[e["entity_norm"]] = e["entity_id"]
        by_friendly.setdefault(e["friendly_norm"], []).append(e["entity_id"])

    payload = {
        "generated_at": time.time(),
        "ha_url": settings.ha_url,
        "counts": {
            "lights": sum(1 for e in entities if e["domain"] == "light"),
            "switches": sum(1 for e in entities if e["domain"] == "switch"),
            "total": len(entities),
        },
        "entities": entities,
        "index": {
            "by_friendly_norm": by_friendly,
            "by_entity_norm": by_entity,
        },
    }

    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(out_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[ha_registry_update] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
