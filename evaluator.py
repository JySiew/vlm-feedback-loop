"""Run the evaluations in a JSON file on the arm and save the evidence."""

import json
import re
import sys
import termios
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from PIL import Image

from utils.VlmModel import OBJECT_SKILLS, VlmModel

ENTRY_KEYS = {
    "name",
    "retry_limit",
    "request",
    "initial_state",
    "expected_intent",
    "expected_final_state",
    "expected_status",
    "event",
}


def _check(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_count(value) -> bool:
    return type(value) is int and value >= 0


def _check_objects(state: dict, count_key: str, label: str) -> set:
    """Validate an initial or final state and return its object names."""
    items = state["objects"]
    names = [item["name"] for item in items]
    _check(len(set(names)) == len(names), f"{label}: repeated object")
    _check(set(names) <= set(OBJECT_SKILLS), f"{label}: unknown object in {names}")
    for item in items:
        _check(set(item) == {"name", count_key}, f"{label}: bad object {item}")
        _check(_is_count(item[count_key]), f"{label}: bad {count_key} in {item}")
    return set(names)


def _check_entry(entry: dict) -> None:
    label = f"Evaluation {entry.get('name')!r}"
    # Unknown keys fail too, so a typo cannot silently disable a check.
    _check(set(entry) == ENTRY_KEYS, f"{label}: keys must be {sorted(ENTRY_KEYS)}")
    name_ok = re.fullmatch(r"[A-Za-z0-9_-]+", entry["name"])
    _check(name_ok, f"{label}: name may use only letters, digits, _ and -")
    limit_ok = _is_count(entry["retry_limit"]) and entry["retry_limit"] >= 1
    _check(limit_ok, f"{label}: retry_limit must be an integer of at least 1")
    _check(entry["request"].strip(), f"{label}: request is empty")
    statuses = {"completed", "failed", "limit_reached"}
    _check(entry["expected_status"] in statuses, f"{label}: bad expected_status")
    _check_objects(entry["initial_state"], "count", label)
    final_objects = _check_objects(entry["expected_final_state"], "stored", label)
    goals = VlmModel.parse_intent(json.dumps({"goals": entry["expected_intent"]}))
    goal_objects = {goal["object"] for goal in goals}
    _check(goal_objects <= final_objects, f"{label}: final state must list all goals")


def load_evaluations(path: Path) -> list[dict]:
    """Return validated evaluations with their defaults filled in."""
    data = json.loads(path.read_text(encoding="utf-8"))
    _check(
        set(data) == {"retry_limit", "evaluations"},
        "File needs exactly retry_limit and evaluations",
    )
    limit = data["retry_limit"]
    _check(_is_count(limit) and limit >= 1, "retry_limit must be at least 1")
    _check(data["evaluations"], "evaluations is empty")
    defaults = {"retry_limit": limit, "expected_status": "completed", "event": None}
    evaluations = [defaults | entry for entry in data["evaluations"]]
    names = [entry["name"] for entry in evaluations]
    _check(len(set(names)) == len(names), "Evaluation names must be unique")
    for entry in evaluations:
        _check_entry(entry)
    return evaluations


def create_run_dir(root: Path, now: datetime) -> Path:
    """Create eval_<number>_<date_time> using the next free number under root."""
    root.mkdir(parents=True, exist_ok=True)
    matches = [re.fullmatch(r"eval_(\d+)_.*", path.name) for path in root.iterdir()]
    number = max((int(match[1]) for match in matches if match), default=0) + 1
    run_dir = root / f"eval_{number}_{now:%Y%m%d_%H%M%S}"
    run_dir.mkdir()
    return run_dir


class TrialRecorder:
    """Save one trial's evidence as it happens, so a crash keeps what is written."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir()

    def on_capture(self, index: int, image) -> None:
        name = f"state_{index}.png" if index else "initial.png"
        Image.fromarray(image).save(self.directory / name)

    def on_batch(self, entry: dict) -> None:
        self.append(entry)

    def append(self, record: dict) -> None:
        path = self.directory / "results.jsonl"
        with path.open("a", encoding="utf-8") as results:
            results.write(json.dumps(record, ensure_ascii=False) + "\n")


def score_trial(evaluation: dict, result: dict) -> dict:
    """
    Compare a loop result with the expected output.
    Unreported objects count as 0 stored, so missing data cannot pass, and
    nothing can be stored if the arm never moved.
    """
    evaluated = [entry for entry in result["history"] if "evaluation" in entry]
    progress = evaluated[-1]["evaluation"]["progress"] if evaluated else []
    stored = {item["object"]: item["stored"] for item in progress}
    expected = evaluation["expected_final_state"]["objects"]
    arm_moved = result["steps"] > 0 or not any(item["stored"] for item in expected)

    def goals(items):
        return sorted(items, key=lambda goal: goal["object"])

    intent = result["intent"]
    return {
        "intent_match": intent is not None
        and goals(intent) == goals(evaluation["expected_intent"]),
        "status_match": result["status"] == evaluation["expected_status"],
        "final_state_match": arm_moved
        and all(stored.get(item["name"], 0) == item["stored"] for item in expected),
    }


def discard_typed_input() -> None:
    """Drop keys pressed during a trial, so a stray Enter cannot start the arm."""
    if sys.stdin.isatty():
        termios.tcflush(sys.stdin, termios.TCIFLUSH)


def _ask_operator(evaluation: dict, input_fn: Callable[[str], str]) -> str:
    """Show the setup and return "" to run, "s" to skip or "q" to quit."""
    setup = ", ".join(
        f"{item['count']} {item['name']}"
        for item in evaluation["initial_state"]["objects"]
    )
    print(f"\n=== {evaluation['name']} (retry limit {evaluation['retry_limit']}) ===")
    print(f"Request: {evaluation['request']}")
    print(f"Table setup: {setup}")
    while True:
        discard_typed_input()
        answer = input_fn("Arm at rest? Enter to run, s to skip, q to quit: ")
        if answer == "" or answer.strip().lower() in {"s", "q"}:
            return answer.strip().lower()


def _run_trial(evaluation: dict, model, executor, run_loop, model_name, run_dir) -> str:
    """Run one evaluation, save its result line, and return its summary label."""
    recorder = TrialRecorder(run_dir / evaluation["name"])
    result = run_loop(
        model,
        executor,
        request=evaluation["request"],
        max_batches=evaluation["retry_limit"],
        observer=recorder,
    )
    scores = score_trial(evaluation, result)
    passed = all(scores.values())
    recorder.append(
        {
            "request": evaluation["request"],
            "retry_limit": evaluation["retry_limit"],
            "model": model_name,
            "status": result["status"],
            "batches": result["batches"],
            "steps": result["steps"],
            "reason": result["reason"],
            "intent": result["intent"],
            **scores,
            "pass": passed,
        }
    )
    label = "PASS" if passed else f"FAIL ({result['status']})"
    print(f"Result: {label}. {result['reason']}")
    return label


def run_session(
    evaluations: list[dict],
    model,
    executors: list,
    run_loop: Callable[..., dict],
    model_name: str,
    output_root: Path,
    input_fn: Callable[[str], str] = input,
) -> int:
    """
    Run the evaluations in order, each with its matching executor.
    Return 0 only if all ran and passed.
    """
    run_dir = create_run_dir(output_root, datetime.now())
    print(f"Saving evidence to {run_dir}")
    outcomes = {entry["name"]: "NOT RUN" for entry in evaluations}
    for evaluation, executor in zip(evaluations, executors):
        action = _ask_operator(evaluation, input_fn)
        if action == "q":
            break
        if action == "s":
            outcomes[evaluation["name"]] = "SKIPPED"
            continue
        outcomes[evaluation["name"]] = _run_trial(
            evaluation, model, executor, run_loop, model_name, run_dir
        )

    print("\nEvaluation summary:")
    for name, outcome in outcomes.items():
        print(f"  {name}: {outcome}")
    passed = sum(outcome == "PASS" for outcome in outcomes.values())
    print(f"Passed {passed} of {len(outcomes)}. Check results against the images.")
    return 0 if passed == len(outcomes) else 1
