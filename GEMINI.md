# TRIDENT — agent context

Marine pollution intelligence for SDG 14, built for a hackathon judged on
Innovation 25%, Technical Architecture 30%, UI/UX 25%, Feasibility & Impact 20%.

Read `README.md` for what the system does and `HANDOFF.md` for setup, the
two-person split, and the current task list. This file is the working
agreement: how to change things here.

## The one-line framing

Everyone else builds "upload → detect → report". TRIDENT is a closed loop:

```
DETECT → LOCALISE → ATTRIBUTE → FORECAST → DISPATCH → VERIFY
```

Every decision should protect that loop. If a change makes the demo tell a
weaker story, it is the wrong change even if the code is nicer.

## Layout

```
api/          FastAPI backend, Python 3.13, venv at api/.venv
  trident/taxonomy/   classes.yaml — SINGLE SOURCE OF TRUTH
  trident/geo/        camera pose → water-plane projection
  trident/drift/      Lagrangian advection on live Open-Meteo forcing
  trident/severity/   the MPSI index
  trident/vision/     YOLO instance branch + physics branch + confuser rejection
  trident/sar/        DeepLabv3+ oil segmenter (vendored from SAMUDRA)
  trident/mission/    clustering, routing, approval gates, vehicle simulator
  trident/api/        HTTP + WebSocket
  scripts/            train_oil.py, eval_oil.py, fetch_wheel.py
  tests/              pytest, 55 tests
web/          Vite + React + TypeScript, MapLibre
```

## Non-negotiables

**1. The taxonomy is the single source of truth.**
`api/trident/taxonomy/classes.yaml` defines every pollution class and its
physics and policy: persistence, mass, windage, harm dimensions, whether a
vehicle may collect it without human sign-off, scrap value. Severity weights,
drift behaviour and autonomy gates all read from it.

Never hardcode a class list, a hazard weight, or a windage value anywhere else.
Retuning the system must stay a data edit.

**2. Confusers are detected so they can be rejected.**
Sea foam, sun glint, vessel wakes, Sargassum and shallow seabed are scored
alongside real pollution and reported as explicitly ruled out, with the
discriminating reason. On open water, natural features are mistaken for
pollution far more often than pollution is missed. Never quietly drop a
candidate — a severity score is only credible next to what it declined to
count.

**3. Uncertainty travels with the number.**
The SAR checkpoint carries a `representative` flag from training all the way to
the UI, because a smoke-test checkpoint still draws a confident outline.
Georeferenced detections carry a positional uncertainty radius. The severity
index carries a confidence band. Keep this habit — it is most of why the
project reads as serious rather than as a demo.

**4. Never present a synthesised value as a measurement.**
Dataset photos have no camera pose, so the operator declares one and it is
labelled `declared` everywhere downstream. Offline drift falls back to a
synthetic field and is labelled synthetic. Do not smooth these away.

**5. Optical and SAR stay separate.**
They are different sensing modalities, not two opinions on the same pixels.
Optical reads an iridescent film in visible light; SAR reads how oil flattens
the sea surface and kills radar backscatter. Feeding a photograph to a model
trained on backscatter returns a confident mask of nothing.

## Code style

- Comments explain *why*, never *what*. If removing the comment would not
  confuse a future reader, delete it. No multi-paragraph docstrings.
- No defensive code for things that cannot happen. Validate at system
  boundaries (uploads, external APIs), trust internal calls.
- No backwards-compatibility shims. This is a four-day-old repo; change the
  code.
- Prefer editing existing files to creating new ones.
- Python: type hints, dataclasses, `from __future__ import annotations`.
- TypeScript: strict mode, no `any`.

## Before claiming anything works

```bash
cd api && .venv/Scripts/python -m pytest -q      # expect 55 passed
cd web && npx tsc --noEmit                        # expect silence
```

Tests passing is not the same as the feature working. For UI changes, actually
run both servers and click through it. Several real bugs here were only found
by running the app: a source-id collision that silently blanked the whole map,
a `ready` ref that meant map layers never populated, and a water mask that
called a beach 94% water.

## Environment gotchas

- **TLS interception.** This network terminates TLS at an inspecting proxy
  whose CA is in the Windows store but not in certifi. `truststore` is a
  dependency and `trident/__init__.py` injects it into `ssl` at import. If
  something fails with `CERTIFICATE_VERIFY_FAILED`, that is why — do not
  disable verification, make sure `import trident` runs first.
- **No CUDA.** The CUDA torch wheel is 2.6 GB and downloads at 1 MB/s here.
  CPU torch is installed deliberately. Train on Colab.
- **Weights and datasets are gitignored.** Never commit `*.pt`, `node_modules`,
  `.venv`, or `api/artifacts/`.

## Known weak spots — do not paper over these

1. The optical detector is COCO-pretrained, not fine-tuned. It found one object
   across ten real pollution photographs. Fine-tuning on TACO is task A1 and is
   the highest-impact work available.
2. The physics branch thresholds were fitted to synthetic fixtures, not to a
   labelled set.
3. There is no SAR oil checkpoint yet. `/api/sar/status` reports
   `available: false` and the UI says so.
4. Water segmentation is a heuristic; it will mislabel very turbid brown water.
5. Sessions are in-memory and clear on backend restart.

If asked to make the project sound better, improve the project. Do not soften
this list.

## Git

Branch per person, `git pull --rebase origin main` before each session.
`web/src/types.ts` mirrors API response shapes — if you change what an endpoint
returns, update it in the same commit.
