# VLM feedback loop

A closed-loop controller for an SO-101 robot arm that tidies pens and pencils into cups.
A vision-language model (VLM) reads a request, plans which skills to run, checks the
result from an overhead camera, and replans when something went wrong. Each skill is a
trained LeRobot diffusion policy.

## How the loop works

1. **Intent:** the VLM turns the request into per-object goals, once per request. For
   example, "store one pencil" becomes `{"object": "pencil", "mode": "take", "number": 1}`.
2. **Plan:** the VLM picks a batch of skills from the overhead image and the goals.
3. **Execute:** the arm runs every skill in the batch.
4. **Evaluate:** the VLM compares the original and current images and reports progress
   for each goal.
5. **Stop or replan:**
   - `completed` when the evaluation is a confident success with every goal met.
   - `limit_reached` after `MAX_BATCHES` batches without success.
   - `failed` when the replan returns an empty batch, for example because a pen landed
     in the wrong cup and no skill can take it out.
   - Otherwise, replan from the evaluator's feedback and go back to step 3.

A skill error or any other exception ends the loop with `error`.

## Skills and objects

| Skill | Stores | Cup |
|---|---|---|
| `pick_place_black_pen` | one black pen | left |
| `pick_place_red_pen` | one red pen | middle |
| `pick_place_pencil` | one pencil | right |

Each skill runs for 25 seconds, then the arm returns to its home pose so the camera has a
clear view.

## Setup

`utils/Executor.py` imports `torch` and `lerobot` at startup, even in simulator mode, and
the imports it uses need `lerobot` 0.6 or newer, which needs Python 3.12 or newer.
`requirements.txt` doesn't list either, so install them as well:

```bash
pip install -r requirements.txt
pip install torch "lerobot[feetech,diffusion]==0.6.1"
```

Then give the loop access to a model, chosen by `MODEL_TYPE` in `main.py`:

- **`soclaas`** (default) uses `MODEL_NAME`, currently `qwen3.8:27b`. Create an env file
  at `~/Downloads/soclaas.env`, or `.env` in this folder, containing:

  ```text
  SOCLAAS_API_KEY=...
  SOCLAAS_BASE_URL=...
  ```

- **`claude`** uses Claude Haiku 5.5 through the Anthropic SDK, with your Anthropic
  credentials.

## Running

All settings are constants at the top of `main.py`.

### On the real arm

Leave `SIMULATOR_MODE = False`, put the arm in its rest pose, then run:

```bash
python main.py
```

Type a request at the prompt, for example `Put away one pencil and one red pen`, and type
`exit` to quit. Each request is one run of the loop, and the result prints at the end.

The arm settings live in `LeRobotExecutor` in `utils/Executor.py`:

| Setting | Current value |
|---|---|
| Serial port | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B79018706-if00` |
| Robot id | `orange_follower_arm` |
| Cameras | overhead index 0, wrist index 2 |
| Checkpoints | `CHECKPOINT_PATHS`, under `/home/hoonc/outputs/diffusion/` |

### In simulation

Set `SIMULATOR_MODE = True` and run `python main.py`. Instead of using the arm, it
replays every recorded event in `events/` through the real loop and model, then prints a
summary. A simulated skill shows the next recorded image instead of moving anything, so
this checks the VLM stages and the loop logic, not whether the arm could carry out the
plans.

## Recorded events

Each folder in `events/` is one recorded scenario:

| File | Contents |
|---|---|
| `request.txt` | The request |
| `before.*` | Overhead image before anything ran |
| `attempt_<n>.*` | Overhead image after batch `n` |
| `expected.json` | Hand-written labels for the event. Never sent to the model |

## Settings in `main.py`

| Setting | Meaning |
|---|---|
| `MODEL_TYPE` | `soclaas` or `claude` |
| `MODEL_NAME` | Model used with `soclaas` |
| `MAX_BATCHES` | Batches allowed per request |
| `MIN_CONFIDENCE` | Lowest evaluator confidence counted as success |
| `SIMULATOR_MODE` | Replay `events/` instead of using the arm |
| `SKILL_NAMES` | Skills the planner may use |
| `POLICY_*` | Diffusion policy inference settings |

## Files

| Path | Purpose |
|---|---|
| `main.py` | The feedback loop and entry point |
| `utils/VlmModel.py` | Model clients and response checks |
| `utils/Prompts.py` | Prompts for intent, planning, evaluation and replanning |
| `utils/Executor.py` | `LeRobotExecutor` for the arm, `LocalSimulator` for replay |
| `utils/Skills.py` | The list of available skills |
| `events/` | Recorded scenarios for simulation |
