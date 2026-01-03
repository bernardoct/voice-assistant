#!/usr/bin/env python3
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import requests

envfile = Path.home() / ".ha_env"
if envfile.exists():
    for line in envfile.read_text().splitlines():
        if line.startswith("export "):
            k, v = line[len("export "):].split("=", 1)
            os.environ.setdefault(k, v.strip('"'))


HA_URL = os.environ.get("HA_URL", "http://192.168.1.203:8123")
HA_TOKEN = os.environ.get("HA_TOKEN")
OUT_PATH = Path(os.environ.get("HA_REGISTRY_PATH", str(Path.home() / ".cache" / "ha_entities.json")))
TIMEOUT = 10

def norm(s: str) -> str:
    s = s.lower().strip()
    # normalize punctuation/spaces
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def ha_get(path: str) -> Any:
    if not HA_TOKEN:
        raise RuntimeError("HA_TOKEN env var not set")
    r = requests.get(
        f"{HA_URL}{path}",
        headers={"Authorization": f"Bearer {HA_TOKEN}"},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()

def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1) Pull current entity states (includes friendly_name, etc.)
    states = ha_get("/api/states")

    entities: List[Dict[str, Any]] = []
    for st in states:
        entity_id = st.get("entity_id", "")
        if not (entity_id.startswith("light.") or entity_id.startswith("switch.")):
            continue

        attrs = st.get("attributes", {}) or {}
        friendly = attrs.get("friendly_name") or entity_id

        entities.append({
            "entity_id": entity_id,
            "domain": entity_id.split(".", 1)[0],
            "friendly_name": friendly,
            "friendly_norm": norm(str(friendly)),
            "entity_norm": norm(entity_id),
            # Optional fields (sometimes present)
            "device_class": attrs.get("device_class"),
            "supported_color_modes": attrs.get("supported_color_modes"),
        })
        if "brightness" in attrs:
            if "supported_color_modes" not in entities[-1]:
                entities[-1]["supported_color_modes"] = ["brightness"]
            elif "brightness" not in entities[-1]["supported_color_modes"]:
                entities[-1]["supported_color_modes"].append("brightness")
        if "color_temp" in attrs.get("supported_color_modes", []):
            # replace color_temp with color_temp_kelvin for clarity
            scm = entities[-1]["supported_color_modes"]
            scm.remove("color_temp")
            scm.append("color_temp_kelvin")

        # lightbulb original rgb color: [255, 164, 82]

    # 2) Build lookup indexes
    # If two entities share a friendly name, keep a list (you’ll disambiguate by area words later)
    by_friendly: Dict[str, List[str]] = {}
    by_entity: Dict[str, str] = {}
    for e in entities:
        by_entity[e["entity_norm"]] = e["entity_id"]
        by_friendly.setdefault(e["friendly_norm"], []).append(e["entity_id"])

    payload = {
        "generated_at": time.time(),
        "ha_url": HA_URL,
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

    tmp = OUT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(OUT_PATH)
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[ha_registry_update] ERROR: {e}", file=sys.stderr)
        sys.exit(1)

