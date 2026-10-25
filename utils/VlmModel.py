import json
from pathlib import Path

import anthropic
from dotenv import dotenv_values
from openai import OpenAI

from .Prompts import (
    EVALUATOR_PROMPT_LIST,
    INTENT_PROMPT_LIST,
    INTENT_RESPONSE_FORMAT,
    PLAN_RESPONSE_FORMAT,
    PLANNER_PROMPT_LIST,
    RETRY_PLANNER_PROMPT_LIST,
    SKILL_NAMES,
)

# Working profile from vlm_benchmark/per_object_working_profile.json.
DEFAULT_MAX_TOKENS = 2048
THINKING_DISABLED_MODELS = {"qwen3.8:27b"}
OBJECT_SKILLS = dict(zip(["pencil", "black_pen", "red_pen"], SKILL_NAMES))
PROGRESS_KEYS = {"object", "required", "stored", "remaining"}


def _is_count(value):
    return type(value) is int and value >= 0


class VlmModel:
    def __init__(self, model_name, api_key, base_url, max_tokens=DEFAULT_MAX_TOKENS):
        self._model_name = model_name
        self._api_key = api_key
        self._base_url = base_url
        self._temperature = 0
        self._top_p = 1
        self._max_tokens = max_tokens
        self._client = OpenAI(
            api_key=api_key, base_url=base_url, timeout=180.0, max_retries=3
        )

    @staticmethod
    def _intent_json(goals):
        return json.dumps({"goals": goals}, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _text(text):
        return {"type": "text", "text": text}

    @staticmethod
    def _image(url):
        return {"type": "image_url", "image_url": {"url": url}}

    def _two_image_content(self, text, original_url, current_url):
        return [
            self._text(text),
            self._text("ORIGINAL TOP"),
            self._image(original_url),
            self._text("CURRENT TOP"),
            self._image(current_url),
        ]

    @staticmethod
    def parse_json(response):
        """
        Parse model response to JSON object
        """
        text = response.strip()
        lines = text.splitlines()
        if (
            len(lines) >= 3
            and lines[0].strip() in {"```json", "```"}
            and lines[-1].strip() == "```"
        ):
            text = "\n".join(lines[1:-1])
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("VLM response must be a JSON object")
        return data

    @classmethod
    def parse_intent(cls, response):
        """Return per-object goals, rejecting malformed or duplicate goals."""
        data = cls.parse_json(response)
        if set(data) != {"goals"} or not isinstance(data["goals"], list):
            raise ValueError("Intent must contain exactly a goals list")
        seen = set()
        for goal in data["goals"]:
            if not isinstance(goal, dict) or set(goal) != {"object", "mode", "number"}:
                raise ValueError(
                    "Each goal must contain exactly object, mode, and number"
                )
            if goal["object"] not in OBJECT_SKILLS:
                raise ValueError(f"Unsupported goal object: {goal['object']!r}")
            if goal["object"] in seen:
                raise ValueError(f"Duplicate goal object: {goal['object']!r}")
            seen.add(goal["object"])
            if goal["mode"] not in {"all", "take", "leave"}:
                raise ValueError(f"Unsupported goal mode: {goal['mode']!r}")
            if not _is_count(goal["number"]):
                raise ValueError("Goal number must be a nonnegative integer")
            if goal["mode"] == "all" and goal["number"] != 0:
                raise ValueError("A goal with mode all must have number 0")
        return data["goals"]

    @classmethod
    def parse_plan(cls, response, goals):
        """Return plan/reason, rejecting unknown skills, unselected types and
        batches that store more than a take,N goal allows."""
        data = cls.parse_json(response)
        if set(data) != {"plan", "reason"} or not isinstance(data["plan"], list):
            raise ValueError("Plan must contain exactly plan and reason")
        if not isinstance(data["reason"], str) or not data["reason"].strip():
            raise ValueError("Plan reason must be a nonempty string")
        selected = {OBJECT_SKILLS[goal["object"]] for goal in goals}
        for skill in data["plan"]:
            if skill not in SKILL_NAMES:
                raise ValueError(f"VLM selected an unsupported skill: {skill!r}")
            if skill not in selected:
                raise ValueError(f"Plan moves an unselected object type: {skill!r}")
        for goal in goals:
            skill = OBJECT_SKILLS[goal["object"]]
            if goal["mode"] == "take" and data["plan"].count(skill) > goal["number"]:
                raise ValueError(
                    f"Plan stores more than {goal['number']} {goal['object']}"
                )
        return data

    @classmethod
    def parse_evaluation(cls, response, goals):
        """Return the progress evaluation after strict validation."""
        data = cls.parse_json(response)
        keys = {"success", "confidence", "progress", "observed", "reason"}
        if set(data) != keys:
            raise ValueError("Evaluation has missing or extra fields")
        if type(data["success"]) is not bool:
            raise ValueError("Evaluation success must be a JSON boolean")
        confidence = data["confidence"]
        if type(confidence) not in {int, float} or not 0 <= confidence <= 1:
            raise ValueError("Evaluation confidence must be a number from 0 to 1")
        for key in ("observed", "reason"):
            if not isinstance(data[key], str) or not data[key].strip():
                raise ValueError(f"Evaluation {key} must be a nonempty string")
        if not isinstance(data["progress"], list):
            raise ValueError("Evaluation progress must be a list")
        for entry in data["progress"]:
            if not isinstance(entry, dict) or set(entry) != PROGRESS_KEYS:
                raise ValueError(
                    "Each progress entry needs object/required/stored/remaining"
                )
            required, stored, remaining = (
                entry["required"],
                entry["stored"],
                entry["remaining"],
            )
            if any(
                v is not None and not _is_count(v)
                for v in (required, stored, remaining)
            ):
                raise ValueError("Progress counts must be nonnegative integers or null")
            if required is not None and stored is not None:
                if remaining != max(required - stored, 0):
                    raise ValueError(f"Inconsistent progress for {entry['object']!r}")
        reported = sorted(entry["object"] for entry in data["progress"])
        if reported != sorted(goal["object"] for goal in goals):
            raise ValueError("Progress must list each selected type exactly once")
        return data

    def infer_intent(self, request):
        """Return the validated per-object goals for the human request."""
        user, system = INTENT_PROMPT_LIST
        text = user + "\n\nRequest: " + json.dumps(request, ensure_ascii=False)
        response = self.infer(
            [self._text(text)],
            system_prompt=system,
            response_format=INTENT_RESPONSE_FORMAT,
        )
        return self.parse_intent(response)

    def plan(self, original_url, goals):
        """Return the validated first action batch for the original top image."""
        user, system = PLANNER_PROMPT_LIST
        content = [
            self._text(user.format(intent=self._intent_json(goals))),
            self._image(original_url),
        ]
        response = self.infer(
            content, system_prompt=system, response_format=PLAN_RESPONSE_FORMAT
        )
        return self.parse_plan(response, goals)

    def replan(self, original_url, current_url, goals, evaluation):
        """Return the validated next action batch using the evaluator's feedback."""
        user, system = RETRY_PLANNER_PROMPT_LIST
        text = user.format(
            intent=self._intent_json(goals),
            feedback=json.dumps(evaluation, ensure_ascii=False),
        )
        response = self.infer(
            self._two_image_content(text, original_url, current_url),
            system_prompt=system,
            response_format=PLAN_RESPONSE_FORMAT,
        )
        return self.parse_plan(response, goals)

    def evaluate(self, original_url, current_url, request, goals):
        """Return the validated progress evaluation (plain output, as benchmarked)."""
        user, system = EVALUATOR_PROMPT_LIST
        text = user.format(request=request, intent=self._intent_json(goals))
        response = self.infer(
            self._two_image_content(text, original_url, current_url),
            system_prompt=system,
        )
        return self.parse_evaluation(response, goals)

    def infer(self, message, image_url=None, system_prompt="", response_format=None):
        """
        Run inference using this instance's configured model.
        """
        if isinstance(message, str):
            content = []
            if image_url:
                content.append(self._image(image_url))
            content.append(self._text(message))
        else:
            if image_url is not None:
                raise ValueError(
                    "Include images as content parts when message is a list"
                )
            content = list(message)
            if not content:
                raise ValueError("message must contain at least one content part")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        options = {}
        if self._model_name in THINKING_DISABLED_MODELS:
            options["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        if response_format is not None:
            options["response_format"] = response_format

        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=self._temperature,
            top_p=self._top_p,
            max_tokens=self._max_tokens,
            **options,
        )
        if not response.choices:
            raise ValueError("Empty VLM response")
        finish_reason = response.choices[0].finish_reason
        if finish_reason != "stop":
            raise ValueError(
                f"Incomplete VLM response: finish_reason={finish_reason!r}"
            )
        return response.choices[0].message.content or ""


class SoCLaaSModel(VlmModel):
    """Load SoCLaaS credentials and endpoint from the selected .env file."""

    def __init__(self, model_name, env_path, max_tokens=DEFAULT_MAX_TOKENS):
        self._env_path = Path(env_path).expanduser()
        api_key, base_url = self._load_env()
        super().__init__(model_name, api_key, base_url, max_tokens=max_tokens)

    def _load_env(self):
        """Read settings without modifying the process environment."""
        if not self._env_path.is_file():
            raise FileNotFoundError(f"Environment file not found: {self._env_path}")
        soclaas_settings = dotenv_values(self._env_path)

        api_key = (soclaas_settings.get("SOCLAAS_API_KEY") or "").strip()
        base_url = (soclaas_settings.get("SOCLAAS_BASE_URL") or "").strip().rstrip("/")
        if not api_key:
            raise ValueError(f"Set SOCLAAS_API_KEY in {self._env_path}.")
        if not base_url:
            raise ValueError(f"Set SOCLAAS_BASE_URL in {self._env_path}.")

        if not base_url.endswith("/v1"):
            base_url += "/v1"
        return api_key, base_url


class ClaudeModel(VlmModel):
    # Structured outputs reject these keywords; the parsers still check values.
    UNSUPPORTED_SCHEMA_KEYS = frozenset({"minimum", "maximum", "minItems", "maxItems"})

    def __init__(self, model_name="claude-haiku-5-5", api_key=None, effort="medium"):
        self._model_name = model_name
        self._max_tokens = 16000
        self._effort = effort
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=3)

    @classmethod
    def _schema(cls, schema):
        if isinstance(schema, dict):
            return {
                k: cls._schema(v)
                for k, v in schema.items()
                if k not in cls.UNSUPPORTED_SCHEMA_KEYS
            }
        if isinstance(schema, list):
            return [cls._schema(item) for item in schema]
        return schema

    @staticmethod
    def _block(part):
        if part["type"] == "text":
            return {"type": "text", "text": part["text"]}
        header, _, data = part["image_url"]["url"].partition(",")
        media_type = header.removeprefix("data:").removesuffix(";base64")
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }

    def infer(self, message, image_url=None, system_prompt="", response_format=None):
        if isinstance(message, str):
            message = [self._text(message)]
            if image_url:
                message.insert(0, self._image(image_url))
        output_config = {"effort": self._effort}
        if response_format is not None:
            schema = self._schema(response_format["json_schema"]["schema"])
            output_config["format"] = {"type": "json_schema", "schema": schema}
        options = {"system": system_prompt} if system_prompt else {}

        response = self._client.beta.messages.create(
            model=self._model_name,
            max_tokens=self._max_tokens,
            thinking={"type": "adaptive"},
            output_config=output_config,
            messages=[{"role": "user", "content": [self._block(p) for p in message]}],
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            **options,
        )
        if response.stop_reason != "end_turn":
            raise ValueError(
                f"Incomplete VLM response: stop_reason={response.stop_reason!r}"
            )
        return "".join(b.text for b in response.content if b.type == "text")
