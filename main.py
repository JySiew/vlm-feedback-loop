import argparse
import base64
import json
from io import BytesIO
from pathlib import Path

from PIL import Image

from evaluator import load_evaluations, run_session
from utils.Executor import ExecStatus, Executor, LeRobotExecutor, LocalSimulator
from utils.Skills import Skills
from utils.VlmModel import OBJECT_SKILLS, ClaudeModel, SoCLaaSModel

# Configuration
PROJECT_DIR = Path(__file__).resolve().parent
_default_env = Path("~/Downloads/soclaas.env").expanduser()
SOCLAAS_ENV_PATH = _default_env if _default_env.is_file() else (PROJECT_DIR / ".env")
MODEL_NAME = "qwen3.8:27b"
MAX_BATCHES = 3
MIN_CONFIDENCE = 0.6
PLANNER_IMAGE_MAX_SIDE = None  # 640
SKILL_NAMES = ["pick_place_pencil", "pick_place_black_pen", "pick_place_red_pen"]
MODEL_TYPE = "soclaas"  # "claude/soclaas"
POLICY_NOISE_SCHEDULER_TYPE = "DDIM"
POLICY_NUM_INFERENCE_STEPS = 5
POLICY_N_ACTION_STEPS = 16


# Mode Enum
class ExecutionMode:
    SIMULATOR = "simulator"
    REAL_ROBOT = "real_robot"
    EVALUATION = "evaluation"
    EVALUATION_SIMULATOR = "evaluation_simulator"


# For Simulator
SIMULATOR_MODE = False
EVENTS_DIR = PROJECT_DIR / "events"
EVALUATION_RESULTS_DIR = PROJECT_DIR / "evaluation_result"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


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


def adjust_plan(pending: list[str], evaluation: dict) -> tuple[list[str], list[str]]:
    """
    Drop skills the evaluator says are no longer needed, keeping the order.
    Returns the kept and removed skills. An unknown count keeps the skill.
    """
    left = {
        OBJECT_SKILLS[item["object"]]: item["remaining"]
        for item in evaluation["progress"]
    }
    kept = []
    removed = []
    for skill in pending:
        if left.get(skill) == 0:
            removed.append(skill)
            continue
        kept.append(skill)
        if left.get(skill):
            left[skill] -= 1
    return kept, removed


def close_executor(executor: Executor) -> None:
    """Close without letting a hardware fault replace a result or a Ctrl-C."""
    try:
        executor.close()
    except Exception as exc:
        print(f"Warning: close failed, check the arm: {type(exc).__name__}: {exc}")


def run_feedback_loop(
    model,
    executor: Executor,
    request: str,
    max_batches: int = MAX_BATCHES,
    observer=None,
) -> dict:
    """
    Intent once, then plan -> execute batch -> evaluate
    Loop is bounded by max_batches.

    Between skills in a batch, the evaluator checks the scene and any skills no
    longer needed are removed from the rest of that batch.

    An optional observer receives on_capture(name, image) straight after each
    capture and on_batch(entry) once per finished batch.
    """
    steps = 0
    batches = 0
    goals = None
    history = []
    debug = {}
    reported = 0

    def captured(name, image):
        if observer is not None:
            observer.on_capture(name, image)

    def report_batches():
        nonlocal reported
        if observer is None:
            return
        while reported < len(history):
            reported += 1
            observer.on_batch(history[reported - 1])

    def outcome(status, reason):
        report_batches()
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
        captured("initial", original)
        original_plan_url = image_to_data_url(original, PLANNER_IMAGE_MAX_SIDE)
        original_eval_url = image_to_data_url(original)
        # Perform intent assessment only once, from the original request
        goals = model.infer_intent(request)
        print(f"Intent: {json.dumps(goals)}")
        plan = check_executable(model.plan(original_plan_url, goals), executor.skills)

        for batch in range(1, max_batches + 1):
            entry = {
                "batch": batch,
                "plan": plan,
                "executed_skills": [],
                "adjustments": [],
            }
            history.append(entry)
            print(f"Batch {batch} plan: {plan['plan']}. {plan['reason']}")
            # Execute the complete batch; an empty plan is still evaluated.
            pending = list(plan["plan"])
            while pending:
                skill_name = pending.pop(0)
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
                if pending:
                    image = executor.peek().overhead
                    captured(f"state_{batch}_step_{steps}", image)
                    check = model.evaluate(
                        original_eval_url, image_to_data_url(image), request, goals
                    )
                    pending, removed = adjust_plan(pending, check)
                    entry["adjustments"].append(
                        {"after_step": steps, "removed": removed, "evaluation": check}
                    )
                    if removed:
                        print(f"Plan adjusted: removed {removed}, next {pending}.")
            batches = batch

            current = executor.capture().overhead
            captured(f"state_{batch}", current)
            evaluation = model.evaluate(
                original_eval_url, image_to_data_url(current), request, goals
            )
            entry["evaluation"] = evaluation
            report_batches()
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
        close_executor(executor)


def make_executor(
    mode: ExecutionMode, skills: Skills, event: dict | None = None
) -> Executor:
    """Build the executor; in simulator mode it replays one event's images."""
    if mode == ExecutionMode.SIMULATOR:
        return LocalSimulator(event["before"], event["after"], event["request"], skills)
    return LeRobotExecutor(
        skills,
        noise_scheduler_type=POLICY_NOISE_SCHEDULER_TYPE,
        num_inference_steps=POLICY_NUM_INFERENCE_STEPS,
        n_action_steps=POLICY_N_ACTION_STEPS,
    )


def simulated_evaluations(
    evaluations: list[dict], skills: Skills
) -> tuple[list[dict], list[Executor]]:
    """Pair each evaluation with a replay of its recorded event."""
    capped = []
    executors = []
    for evaluation in evaluations:
        if evaluation["event"] is None:
            raise ValueError(f"Evaluation {evaluation['name']!r} has no event")
        event = load_event(EVENTS_DIR / evaluation["event"])
        # A recording cannot show more batches than it has images.
        limit = max(1, min(evaluation["retry_limit"], len(event["after"])))
        capped.append({**evaluation, "retry_limit": limit})
        executors.append(make_executor(ExecutionMode.SIMULATOR, skills, event))
    return capped, executors


def report(result: dict) -> None:
    print(
        f"Result: {result['status']} after {result['batches']} batch(es), "
        f"{result['steps']} step(s). {result['reason']}"
    )


def main(
    execution_mode: ExecutionMode = ExecutionMode.REAL_ROBOT, eval_json: str = ""
) -> int:
    Execution_Mode = execution_mode
    executors = []
    try:
        # A bad evaluation file must stop before any client or executor exists.
        simulated = execution_mode == ExecutionMode.EVALUATION_SIMULATOR
        if execution_mode == ExecutionMode.EVALUATION or simulated:
            evaluations = load_evaluations(Path(eval_json))
        # Validate the executor configuration before creating an API client.
        skills = Skills(SKILL_NAMES)
        if execution_mode == ExecutionMode.SIMULATOR:
            events = load_events(EVENTS_DIR)
            executors = [
                make_executor(execution_mode, skills, event) for event in events
            ]
        elif simulated:
            evaluations, executors = simulated_evaluations(evaluations, skills)
        else:
            executors = [make_executor(execution_mode, skills)]
        # Model Initialization
        model = None
        if MODEL_TYPE == "claude":
            model = ClaudeModel()
        elif MODEL_TYPE == "soclaas":
            model = SoCLaaSModel(model_name=MODEL_NAME, env_path=SOCLAAS_ENV_PATH)

        if execution_mode == ExecutionMode.SIMULATOR:
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

        elif execution_mode == ExecutionMode.REAL_ROBOT:
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
        elif execution_mode == ExecutionMode.EVALUATION or simulated:
            return run_session(
                evaluations,
                model,
                executors if simulated else executors * len(evaluations),
                run_feedback_loop,
                str(model._model_name),
                EVALUATION_RESULTS_DIR,
                # Nobody resets a replay, so it never waits for the operator.
                input_fn=(lambda prompt: "") if simulated else input,
            )
    except Exception as exc:
        print(f"Setup failed: {type(exc).__name__}: {exc}")
        return 1
    finally:
        for executor in executors:
            close_executor(executor)


def parse_args(argv: list[str] | None = None) -> tuple[str, str]:
    """Return the execution mode and evaluation file from the command line."""
    parser = argparse.ArgumentParser(description="Run the VLM feedback loop.")
    parser.add_argument("--eval", metavar="PATH", help="run this evaluation file")
    parser.add_argument(
        "--simulator",
        action="store_true",
        help="replay each evaluation's recorded event instead of using the arm",
    )
    args = parser.parse_args(argv)
    if args.simulator and args.eval is None:
        parser.error("--simulator needs --eval")
    if args.eval is None:
        return ExecutionMode.REAL_ROBOT, ""
    if args.simulator:
        return ExecutionMode.EVALUATION_SIMULATOR, args.eval
    return ExecutionMode.EVALUATION, args.eval


if __name__ == "__main__":
    raise SystemExit(main(*parse_args()))
