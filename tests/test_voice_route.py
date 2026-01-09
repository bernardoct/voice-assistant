import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    import requests  # noqa: F401
except ModuleNotFoundError:
    class _RequestsStub:
        class exceptions:
            class RequestException(Exception):
                pass

        def post(self, *args, **kwargs):
            raise NotImplementedError

    sys.modules["requests"] = _RequestsStub()

from assistant_env import load_settings
import voice_route


@pytest.fixture()
def ha_registry():
    fixture_path = "tests/fixtures/ha_registry.json"
    with open(fixture_path, encoding="utf-8") as handle:
        return json.load(handle)


def test_build_prompt_includes_entities_and_rooms(ha_registry):
    prompt = voice_route._build_prompt("turn on the noguchi", ha_registry)
    payload = json.loads(prompt)

    assert payload["action_options"] == ["turn_on", "turn_off", "not_applicable"]
    assert "living_room" in payload["room_options"]
    assert "apartment" in payload["room_options"]

    friendly_names = {item["friendly_name"] for item in payload["entity_options"]}
    assert "Noguchi" in friendly_names
    assert "Bedside Lamp" in friendly_names
    assert "Living Room Crossfade" not in friendly_names
    assert "Humidifier Child lock" not in friendly_names

    noguchi = next(item for item in payload["entity_options"] if item["friendly_name"] == "Noguchi")
    assert noguchi["extra_parameters"] == ["brightness_pct", "color_temp_kelvin"]


def test_llm_request_posts_payload_and_returns_content(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "{\"service\": \"turn_on\"}"}}]}

    def fake_post(url, headers=None, json=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr(voice_route.requests, "post", fake_post)

    content = voice_route._llm_request(
        "http://llm.local", "test-model", "secret-key", "PROMPT"
    )

    assert captured["url"] == "http://llm.local"
    assert captured["headers"]["Authorization"] == "Bearer secret-key"
    assert captured["json"]["model"] == "test-model"
    assert captured["json"]["messages"][0]["content"] == "PROMPT"
    assert content == "{\"service\": \"turn_on\"}"


def test_llm_request_raises_on_missing_choices(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": []}

    monkeypatch.setattr(voice_route.requests, "post", lambda *args, **kwargs: FakeResponse())

    with pytest.raises(RuntimeError, match="missing choices"):
        voice_route._llm_request("http://llm.local", "model", "", "PROMPT")


def test_llm_request_raises_on_missing_content(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": ""}}]}

    monkeypatch.setattr(voice_route.requests, "post", lambda *args, **kwargs: FakeResponse())

    with pytest.raises(RuntimeError, match="missing content"):
        voice_route._llm_request("http://llm.local", "model", "", "PROMPT")


def test_llm_route_builds_prompt_and_parses_response(ha_registry, monkeypatch):
    captured = {}

    def fake_build_prompt(user_text, reg):
        captured["user_text"] = user_text
        captured["reg"] = reg
        return "PROMPT"

    def fake_llm_request(url, model, key, prompt):
        captured["prompt"] = prompt
        captured["url"] = url
        captured["model"] = model
        captured["key"] = key
        return (
            "{\"service\": \"turn_on\", "
            "\"entities\": ["
            "\"light.artemide_tolomeo_mega_living_room_floor_lamp\", "
            "\"light.dresser_lamp\""
            "]}"
        )

    monkeypatch.setattr(voice_route, "_build_prompt", fake_build_prompt)
    monkeypatch.setattr(voice_route, "_llm_request", fake_llm_request)

    settings = SimpleNamespace(
        llm_url="http://llm.local",
        llm_model="test-model",
        llm_api_key="secret-key",
    )

    result = voice_route.llm_route(
        settings, "Please turn on the lights in the living room", ha_registry
    )

    assert captured["user_text"] == "Please turn on the lights in the living room"
    assert captured["reg"] == ha_registry
    assert captured["prompt"] == "PROMPT"
    assert result == {
        "service": "turn_on",
        "entities": [
            "light.artemide_tolomeo_mega_living_room_floor_lamp",
            "light.dresser_lamp",
        ],
    }


def test_llm_request_live(ha_registry):
    prompt = """
        {
        "task": "Select the best Home Assistant action and target from the options and return JSON only. This is the 
        transcription of a verbal command and the STT algorithm may misunderstand words, so be mindful of words that 
        sound similar. If the user asks to control all lights in a room, set room_name from room_options. If the 
        request is unclear or unrelated to home control, set \"service\" to \"not_applicable\". ANSWER IN ENGLISH",
        "user_text": "Please turn on the lights in the living room",
        "action_options": [
            "turn_on",
            "turn_off",
            "not_applicable"
        ],
        "entity_options": [
            {
            "friendly_name": "Artemide Tolomeo Mega (Living Room Floor Lamp)",
            "extra_parameters": []
            },
            {
            "friendly_name": "Countertop light",
            "extra_parameters": []
            },
            {
            "friendly_name": "Bedside Lamp",
            "extra_parameters": [
                "brightness_pct"
            ]
            },
            {
            "friendly_name": "Noguchi",
            "extra_parameters": [
                "brightness_pct",
                "color_temp_kelvin"
            ]
            },
            {
            "friendly_name": "Office floor lamp",
            "extra_parameters": [
                "brightness_pct",
                "color_temp_kelvin"
            ]
            },
            {
            "friendly_name": "Humidifier",
            "extra_parameters": []
            },
            {
            "friendly_name": "Countertop light",
            "extra_parameters": []
            },
            {
            "friendly_name": "Artemide Tolomeo Mega (Living Room Floor Lamp)",
            "extra_parameters": []
            }
        ],

        "areas": [
            {
            "light_entities": [
                "light.artemide_tolomeo_mega_living_room_floor_lamp",
                "light.dresser_lamp"
            ],
            "name": "living_room"
            },
            {
            "light_entities": [
                "light.office_floor_lamp"
            ],
            "name": "office"
            },
            {
            "light_entities": [
                "light.countertop_light"
            ],
            "name": "kitchen"
            },
            {
            "light_entities": [
                "light.bedside_lamp"
            ],
            "name": "bedroom"
            },
            {
            "light_entities": [],
            "name": "apartment"
            }
        ],
        "output_schema": {
            "service": "string, one of action_options; required when intent is 'action'",
            "entity_friendly_name": "string, one of entity_options.friendly_name; omit if the user targeted an entire room rather than one entity in that room.",
            "room_name": "string, one of room_options; use ONLY when the user targeted all lights in an ENTIRE room, otherwise omit",
            "data": "object; optional. Omit if not requested in user_text. Only include keys from selected entity_options.extra_parameters. THERE MAY BE MULTIPLE."
        }
        }
        """

    settings = load_settings()
    
    text = voice_route._llm_request(settings.llm_url, settings.llm_model, settings.llm_api_key, prompt)
    response = json.loads(text)

def test_llm_route_builds_prompt_and_parses_response_live(ha_registry):
    captured = {}

    settings = load_settings()

    result = voice_route.llm_route(
        settings, "Please turn on the lights in the living room", ha_registry
    )

    assert captured["user_text"] == "Please turn on the lights in the living room"
    assert captured["reg"] == ha_registry
    assert captured["prompt"] == prompt
    assert result == {
        "service": "turn_on",
        "entities": [
            "light.artemide_tolomeo_mega_living_room_floor_lamp",
            "light.dresser_lamp",
        ],
    }
