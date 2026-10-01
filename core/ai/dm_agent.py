"""The AI Dungeon Master.

Wraps OpenAI, Anthropic, or a local Ollama model behind one ``respond()`` call.
Every turn combines what the player said, where every miniature stands, the
tracked game state, and (optionally) an overhead camera snapshot. The model must
answer with a fixed JSON shape so the projector can animate its effects.
"""

from __future__ import annotations

import base64
import json
import os
import re
import threading

import requests

SYSTEM_PROMPT = """You are the Dungeon Master for a Dungeons & Dragons 5th Edition game played on a real table.
Players move physical miniatures on a 1-inch grid; a camera tracks them and a projector draws your effects onto the table.

Rules:
- Run the game by the 5e rules: roll dice yourself when needed (say the roll in the narration), track HP, conditions and initiative.
- Be vivid but brief: narration is read aloud, so keep it to 1-4 sentences unless a scene truly needs more.
- Positions you receive are authoritative. Use grid coordinates for range and area of effect (1 square = 5 ft).
- If a player asks for something impossible from where their miniature stands, say so in the fiction.
- You may ask the players to move monster miniatures for you.

Reply with ONLY a JSON object, no markdown, in exactly this shape:
{
  "narration": "what you say out loud",
  "grid_effects": [
    {"type": "spell_aura", "x": 5, "y": 8, "radius": 2, "color": "gold", "duration": 6}
  ],
  "state_updates": {"goblin_1_hp": 12}
}
- grid_effects: visual effects to project. x/y are grid squares, radius in squares.
  type is one of: spell_aura, area_burst, highlight, line, darkness. color is a CSS color name or hex.
  For "line" also give "x2" and "y2". Use [] when nothing should be drawn.
- state_updates: only the keys that changed (hp, conditions, initiative order, quest flags...). Use {} if nothing changed.
"""

FALLBACK = {
    "narration": "The Dungeon Master pauses, lost in thought. Try that again in a moment.",
    "grid_effects": [],
    "state_updates": {},
}

EFFECT_TYPES = {"spell_aura", "area_burst", "highlight", "line", "darkness"}


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (tolerates code fences / chatter)."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        return json.loads(fenced.group(1))
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in reply")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    return obj


def _num(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clean_response(raw: dict) -> dict:
    """Coerce a model reply into the exact schema the frontend expects."""
    if not isinstance(raw, dict):
        raise ValueError("reply is not an object")
    effects = []
    for e in raw.get("grid_effects") or []:
        if not isinstance(e, dict):
            continue
        x, y = _num(e.get("x")), _num(e.get("y"))
        if x is None or y is None:
            continue
        effect = {
            "type": e.get("type") if e.get("type") in EFFECT_TYPES else "spell_aura",
            "x": x,
            "y": y,
            "radius": max(0.5, _num(e.get("radius"), 1.0)),
            "color": str(e.get("color") or "gold"),
            "duration": min(60.0, max(1.0, _num(e.get("duration"), 6.0))),
        }
        if effect["type"] == "line":
            effect["x2"], effect["y2"] = _num(e.get("x2"), x), _num(e.get("y2"), y)
        effects.append(effect)
    updates = raw.get("state_updates")
    return {
        "narration": str(raw.get("narration") or "").strip(),
        "grid_effects": effects,
        "state_updates": updates if isinstance(updates, dict) else {},
    }


def _require_key(env_name: str) -> str:
    key = os.environ.get(env_name)
    if not key:
        raise RuntimeError(f"Set the {env_name} environment variable (keys are never read from config files).")
    return key


class DMAgent:
    def __init__(self, config: dict):
        self.cfg = config["ai"]
        self.provider = self.cfg["provider"]
        self.history: list[dict] = []  # text-only turns: {"role": "user"|"assistant", "content": str}
        self.game_state: dict = {}
        self._lock = threading.Lock()
        self._client = None
        if self.provider in ("openai", "openrouter"):
            from openai import OpenAI  # OpenRouter speaks the OpenAI API
            p = self.cfg[self.provider]
            self._client = OpenAI(api_key=_require_key(p["api_key_env"]), base_url=p.get("base_url") or None)
        elif self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=_require_key(self.cfg["anthropic"]["api_key_env"]))
        elif self.provider != "ollama":
            raise ValueError(f"Unknown ai.provider '{self.provider}' (use openrouter, anthropic, openai or ollama)")

    # ---------------------------------------------------------------- public

    def respond(self, player_text: str, entities: list[dict], snapshot_jpeg: bytes | None = None) -> dict:
        """One DM turn. Never raises: model failures come back as an in-character pause."""
        with self._lock:
            turn_text = self._turn_text(player_text, entities)
            image = snapshot_jpeg if self.cfg.get("send_snapshot", True) else None
            try:
                reply_text = self._call_model(turn_text, image)
                result = clean_response(extract_json(reply_text))
            except Exception as e:
                print(f"[dm] model call failed: {type(e).__name__}: {e}")
                return dict(FALLBACK)
            self.game_state.update(result["state_updates"])
            self.history += [
                {"role": "user", "content": turn_text},
                {"role": "assistant", "content": json.dumps(result)},
            ]
            keep = 2 * int(self.cfg.get("max_history_turns", 20))
            self.history = self.history[-keep:]
            return result

    def reset(self) -> None:
        with self._lock:
            self.history.clear()
            self.game_state.clear()

    # ---------------------------------------------------------------- prompt

    def _turn_text(self, player_text: str, entities: list[dict]) -> str:
        table = "\n".join(
            f"- {e['label']} ({e['role']}, id {e['entity_id']}) at grid ({e['grid_x']}, {e['grid_y']})"
            for e in sorted(entities, key=lambda e: e["entity_id"])
        ) or "- (no miniatures detected)"
        return (
            f"TABLE (from the overhead camera):\n{table}\n\n"
            f"GAME STATE:\n{json.dumps(self.game_state)}\n\n"
            f"PLAYER SAYS: \"{player_text}\""
        )

    # ---------------------------------------------------------------- providers

    def _call_model(self, turn_text: str, image: bytes | None) -> str:
        b64 = base64.b64encode(image).decode() if image else None
        if self.provider in ("openai", "openrouter"):
            return self._openai(turn_text, b64)
        if self.provider == "anthropic":
            return self._anthropic(turn_text, b64)
        return self._ollama(turn_text, b64)

    def _openai(self, turn_text: str, b64: str | None) -> str:
        content: list[dict] = [{"type": "text", "text": turn_text}]
        if b64:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        extra = {}
        if self.provider == "openai":
            # Not every OpenRouter model supports JSON mode; the prompt + extract_json cover those.
            extra["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(
            model=self.cfg[self.provider]["model"],
            temperature=self.cfg.get("temperature", 0.8),
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, *self.history, {"role": "user", "content": content}],
            **extra,
        )
        return resp.choices[0].message.content or ""

    def _anthropic(self, turn_text: str, b64: str | None) -> str:
        content: list[dict] = []
        if b64:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}})
        content.append({"type": "text", "text": turn_text})
        p = self.cfg["anthropic"]
        resp = self._client.messages.create(
            model=p["model"],
            max_tokens=p.get("max_tokens", 1024),
            temperature=self.cfg.get("temperature", 0.8),
            system=SYSTEM_PROMPT,
            messages=[*self.history, {"role": "user", "content": content}],
        )
        return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")

    def _ollama(self, turn_text: str, b64: str | None) -> str:
        p = self.cfg["ollama"]
        user = {"role": "user", "content": turn_text}
        if b64:
            user["images"] = [b64]
        resp = requests.post(
            f"{p['host'].rstrip('/')}/api/chat",
            json={
                "model": p["model"],
                "stream": False,
                "format": "json",
                "options": {"temperature": self.cfg.get("temperature", 0.8)},
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *self.history, user],
            },
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
