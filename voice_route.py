#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_registry(registry_path: str) -> Dict[str, Any]:
    return json.loads(Path(registry_path).read_text())


def _registry_rows(reg: Dict[str, Any]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    mode_to_attribute = {
        "brightness": "brightness",
        "color_temp_kelvin": "warmth",
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
            extra_attributes = [
                mode_to_attribute[m] for m in supported_modes if m in mode_to_attribute
            ]
            entity_id = entity.get("entity_id", "")
            entity_type = entity_id.split(".", 1)[0] if entity_id else "entity"
            rows.append(
                {
                    "item": friendly_name,
                    "type": entity_type,
                    "other_attributes": ", ".join(extra_attributes),
                }
            )
    for area in reg.get("areas", []):
        name = area.get("name")
        if name:
            rows.append(
                {
                    "item": name,
                    "type": "area",
                    "other_attributes": "",
                }
            )
    return rows


_RAG_INDEX: VectorStoreIndex | None = None
_RAG_ROWS: List[Dict[str, str]] | None = None


def _row_text(row: Dict[str, str]) -> str:
    return f"{row.get('item', '')} {row.get('type', '')} {row.get('other_attributes', '')}".strip()


def _get_rag_index(rows: List[Dict[str, str]]) -> VectorStoreIndex:
    global _RAG_INDEX, _RAG_ROWS
    if _RAG_INDEX is None or _RAG_ROWS != rows:
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        docs = [Document(text=_row_text(row), metadata={"row": row}) for row in rows]
        _RAG_INDEX = VectorStoreIndex.from_documents(docs)
        _RAG_ROWS = rows
    return _RAG_INDEX


def _select_relevant_rows(
    user_text: str,
    rows: List[Dict[str, str]],
    max_rows: int = 20,
) -> List[Dict[str, str]]:
    if not rows:
        return []
    index = _get_rag_index(rows)
    retriever = index.as_retriever(similarity_top_k=max_rows)
    nodes = retriever.retrieve(user_text)
    return [node.metadata["row"] for node in nodes if "row" in node.metadata]


def _format_registry_table(rows: List[Dict[str, str]]) -> str:
    header = "| item | type | other attributes |"
    separator = "| --- | --- | --- |"
    lines = [header, separator]
    for row in rows:
        item = row.get("item", "").replace("|", " ")
        row_type = row.get("type", "").replace("|", " ")
        attrs = row.get("other_attributes", "").replace("|", " ")
        lines.append(f"| {item} | {row_type} | {attrs} |")
    return "\n".join(lines)


def _build_prompt(user_text: str, reg: Dict[str, Any]) -> str:
    actions = ["turn_on", "turn_off", "not_applicable"]
    rows = _registry_rows(reg)
    relevant_rows = _select_relevant_rows(user_text, rows)
    registry_table = _format_registry_table(relevant_rows)
    prompt = {
        "task": "Select the best Home Assistant action and target from the options and return JSON only. This is the "
        "transcription of a verbal command and the STT algorithm may misunderstand words, so be mindful of words "
        "that sound similar. If the user asks to control all lights in a room, set room_name using an item from the "
        "table where type is 'area'. "
        "If the request is unclear or unrelated to home control, set \"service\" to \"not_applicable\". "
        # "If the request is unrelated to home control or is unclear, set intent to reply and provide response_text. "
        # "If the request seems like an accidental trigger of the voice assistant, respond with an empty reply."
        "ANSWER IN ENGLISH. The table below lists available targets. The 'other attributes' column uses plain "
        "language; map 'brightness' to brightness_pct and 'warmth' to color_temp_kelvin in the data object.",
        "user_text": user_text,
        "action_options": actions,
        "registry_table": registry_table,
        "output_schema": {
            # "intent": "string, either 'action' or 'reply'",
            "service": "string, one of action_options; required when intent is 'action'",
            "entity_friendly_name": "string, one of registry_table.item where type is not 'area'; omit if targeting an entire room rather than one entity in that room.",
            "room_name": "string, one of registry_table.item where type is 'area'; use ONLY when targeting all lights in an ENTIRE room, otherwise omit",
            "data": "object; optional. Omit if not requested in user_text. Allowed keys: brightness_pct, color_temp_kelvin. THERE MAY BE MULTIPLE.",
            # "response_text": "string; required when intent is 'reply'",
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
