import base64
import json
from io import BytesIO
from pathlib import Path

from PIL import Image

from utils.Executor import ExecStatus, Executor, LeRobotExecutor, LocalSimulator
from utils.Skills import Skills
from utils.VlmModel import ClaudeModel, SoCLaaSModel

# Configuration
PROJECT_DIR = Path(__file__).resolve().parent
SOCLAAS_ENV_PATH = Path("~/Downloads/soclaas.env").expanduser()
MODEL_NAME = "qwen3.8:27b"
MAX_BATCHES = 3
MIN_CONFIDENCE = 0.6
PLANNER_IMAGE_MAX_SIDE = None # 640
SKILL_NAMES = [
    "pick_place_pencil",
    "pick_place_black_pen",
    "pick_place_red_pen"
]
MODEL_TYPE = "soclaas"  # "claude/soclaas"

# For Simulator
SIMULATOR_MODE = True
EVENTS_DIR = PROJECT_DIR / "events"
IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png"
}


def image_to_data_url(image, max_side=None) -> str:
    """
    Load image and resize if available.
    Then, convert to base64 data URL.
    """
    picture = Image.fromarray(image)
    if max_side is not None:
        picture.thumbnail((max_side, max_side))
    buffer = BytesIO()
    picture.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def load_event(event_dir: Path) -> dict:
    """
    Read one event directory for simulation:
    1. request.txt
    2. before.*
    3. attempt_<n>.* images
    """
    request_path = event_dir / "request.txt"
    if not request_path.is_file():
        raise ValueError(f"Event {event_dir.name!r} has no request.txt")
    images = [
        path
        for path in event_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    before = [path for path in images if path.stem == "before"]
    if len(before) != 1:
        raise ValueError(f"Event {event_dir.name!r} needs exactly one before image")
    attempts = {}
    for path in images:
        prefix, _, number = path.stem.partition("_")
        if prefix != "attempt" or not number.isdigit():
            continue
        if int(number) in attempts:
            raise ValueError(f"Event {event_dir.name!r} repeats attempt {number}")
        attempts[int(number)] = path
    if sorted(attempts) != list(range(1, len(attempts) + 1)):
        raise ValueError(
            f"Event {event_dir.name!r} attempts must be numbered 1, 2, ..."
        )
    return {
        "name": event_dir.name,
        "request": request_path.read_text(encoding="utf-8").strip(),
        "before": before[0],
        "after": [attempts[number] for number in sorted(attempts)],
    }


def load_events(events_dir: Path) -> list[dict]:
    """Read every event subdirectory"""
    if not events_dir.is_dir():
        raise FileNotFoundError(f"Events directory not found: {events_dir}")
    events = [
        load_event(path) for path in sorted(events_dir.iterdir()) if path.is_dir()
    ]
    if not events:
        raise ValueError(f"No event subdirectories in {events_dir}")
    return events


def check_executable(plan: dict, skills: Skills) -> dict:
    """Reject planned skills that this executor does not support."""
    for skill in plan["plan"]:
        if skill not in skills.get_skills():
            raise ValueError(f"Executor does not support planned skill: {skill!r}")
    return plan


def passes_success_gate(evaluation: dict, steps: int) -> bool:
    """Confident success and exact known progress per goal."""
    if evaluation["success"] is not True:
        return False
    if evaluation["confidence"] < MIN_CONFIDENCE:
        return False
    for entry in evaluation["progress"]:
        if entry["required"] is None or entry["stored"] != entry["required"]:
            return False
    # No-action success (no goals) also requires that the robot never moved.
    return bool(evaluation["progress"]) or steps == 0


def run_feedback_loop(
    model, executor: Executor, request: str, max_batches: int = MAX_BATCHES
) -> dict:
    """
    Intent once, then plan -> execute batch -> evaluate
    Loop is bounded by max_batches.
    """
    steps = 0
    batches = 0
    goals = None
    history = []

    def outcome(status, reason):
        return {
            "status": status,
            "steps": steps,
            "batches": batches,
            "reason": reason,
            "intent": goals,
            "history": history,
        }

    try:
        # Validate the inputs before starting the loop
        if type(max_batches) is not int or max_batches < 1:
            raise ValueError("max_batches must be a positive integer")
        if not isinstance(request, str) or not request.strip():
            raise ValueError("request must be a nonempty string")
        # Connect to the executor
        executor.connect()
        # Planners get a resized original image; the evaluator gets full size.
        original = executor.capture().overhead
        original_plan_url = image_to_data_url(original, PLANNER_IMAGE_MAX_SIDE)
        original_eval_url = image_to_data_url(original)
        # Perform intent assessment only once, from the original request
        goals = model.infer_intent(request)
        print(f"Intent: {json.dumps(goals)}")
        plan = check_executable(model.plan(original_plan_url, goals), executor.skills)

        for batch in range(1, max_batches + 1):
            entry = {"batch": batch, "plan": plan, "executed_skills": []}
            history.append(entry)
            print(f"Batch {batch} plan: {plan['plan']}. {plan['reason']}")
            # Execute the complete batch; an empty plan is still evaluated.
            for skill_name in plan["plan"]:
                result = executor.run(skill_name)
                entry["execution_status"] = result.status.value
                # A timeout may still have succeeded slowly, so it is evaluated.
                if result.status not in {ExecStatus.COMPLETED, ExecStatus.TIMEOUT}:
                    return outcome(
                        "error",
                        result.error or f"Execution returned {result.status.value}",
                    )
                entry["executed_skills"].append(skill_name)
                steps += 1
                print(f"Step {steps}: executed {skill_name} ({result.status.value}).")
            batches = batch

            current = executor.capture().overhead
            evaluation = model.evaluate(
                original_eval_url, image_to_data_url(current), request, goals
            )
            entry["evaluation"] = evaluation
            print(f"Evaluation {batch}: {evaluation['reason']}")

            if passes_success_gate(evaluation, steps):
                return outcome("completed", evaluation["reason"])
            if batch == max_batches:
                return outcome(
                    "limit_reached",
                    f"Reached the limit of {max_batches} batches without "
                    f"confirmed completion. {evaluation['reason']}",
                )

            current_url = image_to_data_url(current, PLANNER_IMAGE_MAX_SIDE)
            plan = check_executable(
                model.replan(original_plan_url, current_url, goals, evaluation),
                executor.skills,
            )
            if not plan["plan"]:
                return outcome("failed", f"No supported batch: {plan['reason']}")
    except Exception as exc:
        return outcome("error", f"{type(exc).__name__}: {exc}")
    finally:
        executor.close()


def make_executor(skills: Skills, event: dict | None = None) -> Executor:
    """Build the executor; in simulator mode it replays one event's images."""
    if SIMULATOR_MODE:
        return LocalSimulator(event["before"], event["after"], event["request"], skills)
    # CHANGE_ME: REPLLACE THIS WITH REAL ROBOT EXECUTOR CLASS with proper initialization
    return LeRobotExecutor(skills)


def report(result: dict) -> None:
    print(
        f"Result: {result['status']} after {result['batches']} batch(es), "
        f"{result['steps']} step(s). {result['reason']}"
    )


def main() -> int:
    executors = []
    try:
        # Validate the executor configuration before creating an API client.
        skills = Skills(SKILL_NAMES)
        if SIMULATOR_MODE:
            events = load_events(EVENTS_DIR)
            executors = [make_executor(skills, event) for event in events]
        else:
            executors = [make_executor(skills)]
        # Model Initialization
        model = None
        if MODEL_TYPE == "claude":
            model = ClaudeModel()
        elif MODEL_TYPE == "soclaas":
            model = SoCLaaSModel(model_name=MODEL_NAME, env_path=SOCLAAS_ENV_PATH)

        if SIMULATOR_MODE:
            # Each loop is one recorded event with its own request and images.
            statuses = {}
            for event, executor in zip(events, executors):
                print(f"\n=== Event {event['name']}: {event['request']} ===")
                # A recording cannot show more batches than it has images.
                max_batches = max(1, min(MAX_BATCHES, len(event["after"])))
                result = run_feedback_loop(
                    model, executor, request=event["request"], max_batches=max_batches
                )
                report(result)
                statuses[event["name"]] = result["status"]
            print("\nSummary:")
            for name, status in statuses.items():
                print(f"  {name}: {status}")
            return 0 if all(s == "completed" for s in statuses.values()) else 1

        while True:
            # Ask for a user request
            user_request = input("Enter a request (or 'exit' to quit): ").strip()
            if user_request.lower() == "exit":
                print("Exiting the feedback loop.")
                return 0
            result = run_feedback_loop(
                model, executors[0], request=user_request, max_batches=MAX_BATCHES
            )
            report(result)
    except Exception as exc:
        print(f"Setup failed: {type(exc).__name__}: {exc}")
        return 1
    finally:
        for executor in executors:
            executor.close()


if __name__ == "__main__":
    raise SystemExit(main())
