# TRIDENT — handoff and two-person split

Written mid-build so a second person can pick up cold. Read `README.md` first
for what the system does; this file is about who does what next.

---

## Setting up on a second machine

`.venv/`, `node_modules/`, model weights, datasets and sample images are all
gitignored, so a fresh clone is code only. Everything below is needed once.

### 1. Clone

```bash
git clone https://github.com/ALEN4343/trident.git
cd trident
```

### 2. Backend

Needs Python 3.11+ (3.13 is what this was built on) and Node 20+.

```bash
cd api
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install ultralytics opencv-python-headless scikit-image
.venv/Scripts/python -m pytest -q
```

Expect 55 passing tests. CPU torch is deliberate — the CUDA wheel is 2.6 GB and
will not download on the college network. Train on Colab instead.

> **If anything fails with `CERTIFICATE_VERIFY_FAILED`:** the campus network
> terminates TLS at an inspecting proxy whose root certificate is in the
> Windows store but not in certifi. `truststore` is already a dependency and
> `trident/__init__.py` installs it into `ssl` at import, which fixes every
> library that fetches something. If `pip` itself fails, add
> `--trusted-host pypi.org --trusted-host files.pythonhosted.org`.

### 3. Model weights

The YOLO checkpoint downloads itself on first inference (about 6 MB) as long as
`import trident` has run, which every entry point does. Nothing to do manually.

The SAR oil checkpoint does **not** exist yet — that is task A2. Until it is
trained, `/api/sar/status` reports `available: false` and the UI says so
instead of breaking.

### 4. Frontend

```bash
cd web
npm install
npm run dev
```

Vite proxies `/api` and `/ws` to `127.0.0.1:8000`, so run the backend too.

### 5. Test imagery

Not in the repo. Grab a handful of water-body photographs — the pipeline runs
on anything, and the dropzone accepts drag and drop. For the SAR mode you need
the SOS dataset, which currently exists only on Alen's machine
(1.4 GB, `DESKTOP_2/SIH_Demo/data/raw/oilspill/`). Copy it across on a USB
stick or share the three demo scenes in `web/public/sar/` directly.

---

## Working on two machines without stepping on each other

The split below is by directory precisely so this stays simple.

```bash
git checkout -b your-name/what-you-are-doing
# work, commit
git push -u origin your-name/what-you-are-doing
```

Then open a PR, or merge to `main` directly if you are moving fast — with two
people in separate directories, direct-to-main is usually fine.

**Before you start each session:**

```bash
git pull --rebase origin main
```

**The one file you will both want to touch** is `web/src/types.ts`, which
mirrors the API response shapes. If Person A changes what an endpoint returns,
Person A updates `types.ts` in the same commit and tells Person B. That is the
whole protocol.

**Do not commit:** `.venv/`, `node_modules/`, `*.pt` weights, datasets,
`api/artifacts/`. All already gitignored — if `git status` shows any of them,
something is wrong with the ignore file rather than with you.

---

## Where things actually stand

**Working end to end.** Upload an image → detections with per-object lat/lon →
severity score → drift forecast → cleanup mission → approval gates → simulated
vessel flies it. 55 backend tests pass, frontend typechecks clean.

**Two servers:**

```bash
cd api && .venv/Scripts/python -m uvicorn trident.api.main:app --port 8000
```

```bash
cd web && npm run dev
```

Then http://localhost:5173.

### What is genuinely strong

- Per-object georectification (pinhole ray → water-plane intersection). Gives
  real coordinates, ground sample distance, object size in cm, imaged water
  area. This is the technical centrepiece.
- Drift forecasting on live Open-Meteo currents, forward and reverse. Windage
  comes from the taxonomy, so a foam block and a waterlogged net separate.
- Mission routing against *predicted* positions, not observed ones.
- Confuser rejection — foam, glint, wake, Sargassum reported as explicitly
  ruled out, with the discriminating reason.
- Human approval gates with a full audit log.

### What is weak, and you should know before a judge finds out

1. **The optical object detector is COCO-pretrained, not fine-tuned.** On ten
   real pollution photographs it found one object. It reliably detects bottles,
   cups and glassware at close range; it misses small litter in wide aerial
   shots, which is most real survey imagery. The physics branch carries the
   demo today.
2. **The physics branch is calibrated on synthetic scenes.** Oil/foam/glint
   separation behaves correctly on constructed fixtures and on some real
   photos, but thresholds were never fitted to a labelled set.
3. **The SAR oil checkpoint is CPU-trained on a subset.** It marks itself
   `representative: false` and the UI surfaces that. Do not quote its numbers
   as performance.
4. Water segmentation is a heuristic. Handles sky, vegetation and dry sand;
   will mislabel very turbid brown water.
5. Sessions are in memory — restarting the backend clears them.

### Blocked

CUDA torch will not install on this network. `pip` resumes from the same offset
without advancing, `curl` cannot complete the TLS handshake to the host, and
the one path that works (Python + OS trust store) runs at 1.0 MB/s against a
2.6 GB wheel. **Train on Colab instead** — `notebooks/train_colab.ipynb` exists
in the SAMUDRA project and the dataset is only 1.4 GB.

---

## The split

Divided by directory so the two of you almost never touch the same file.
`api/` and `web/` are independent; the contract between them is
`web/src/types.ts`, which mirrors the API responses. **If you change an API
response shape, update `types.ts` in the same commit.**

### Person A — models and backend (`api/`)

Owns everything under `api/`. The theme is *make the detections real*.

**A1. Fine-tune the optical detector.** Highest impact by far — it is the
weakest link and the most visible.
- Dataset: TACO (`tacodataset.org`, 1500 images, 60 classes, COCO format),
  optionally TrashCan 1.0.
- The class mapping already exists: `COCO_TO_TAXONOMY` in
  `api/trident/vision/detector.py`. Extend it to TACO's taxonomy.
- Train YOLOv11-seg on Colab, drop the weights into `api/weights/`, and point
  `DEFAULT_WEIGHTS` at them. Nothing else has to change.
- Augment hard for water: specular glare, blue-green channel attenuation,
  motion blur. This is the difference between 60% and 85% mAP on real imagery.

**A2. Finish the SAR oil model.** Train on Colab with the pretrained backbone.
- `python scripts/train_oil.py --epochs 20 --batch-size 8`
- Then `python scripts/eval_oil.py --checkpoint weights/oil_seg.pt`
- Watch the per-sensor gap it prints. If PALSAR and Sentinel-1 IoU differ by
  more than ~0.15 the model learned a sensor, not oil, and the number is not
  trustworthy regardless of how good it looks.

**A3. Calibrate the physics branch.** Thresholds in
`api/trident/vision/indices.py` (`_HYPOTHESES`, `_MIN_HYPOTHESIS_SCORE`) were
set against synthetic fixtures. Label 20–30 real photos and fit them.

**A4. Persist sessions.** SQLite is enough. `SESSIONS` and `MISSIONS` dicts in
`api/trident/api/main.py`.

### Person B — frontend and demo (`web/`)

Owns everything under `web/`, plus the pitch. The theme is *make the story
land in three minutes*.

**B1. Report export.** There is no PDF/report output yet and the problem
statement explicitly asks for "automated pollution report generation". Build a
print-styled route that renders severity breakdown, detections with
coordinates, rejected candidates, drift forecast and mission summary, then
`window.print()` to PDF. Highest-value missing deliverable.

**B2. Demo script and sample imagery.** Pick 3–4 images where the pipeline
visibly shines and script the exact click path. Right now the demo depends on
finding a good image live, which is a risk. Samples live in `web/public/`
(gitignored — copy them to Chris's machine directly).

**B3. Ground truth comparison for SAR.** `web/public/sar/` has three scenes
with `-truth.png` masks. Showing prediction against truth side by side is the
single most convincing thing you can put on screen for the model.

**B4. Polish.** The layout breaks below ~1200px — it is a three-column grid
with fixed rails in `web/src/styles.css`. Add a breakpoint. Also: empty states,
loading states, and a legend explaining the uncertainty circles.

**B5. The pitch.** 25% of the score is innovation and 25% is UI/UX. Lead with
the loop, not the model:

> Everyone else tells you the ocean is dirty. We tell you exactly where each
> piece is, where it came from, where it will be in six hours, and we dispatch
> a boat to intercept it — with a human approving every pickup.

Then show, in order: rejected candidates panel (credibility), per-object
coordinates (technical depth), drift forecast (nobody else has this), mission
gates (safety story).

### Shared / whoever gets there first

- Deploy so judges get a URL. Frontend on Vercel, backend on Railway or Render.
- `api/artifacts/` holds training metrics and plots — useful slide material.

---

## Conventions

- Taxonomy is the single source of truth:
  `api/trident/taxonomy/classes.yaml`. Severity weights, drift windage and
  autonomy policy all read from it. There are no hardcoded class lists
  elsewhere — retuning is a data edit, not a code change.
- Anything a model asserts should carry whether it is trustworthy. The SAR
  checkpoint's `representative` flag travels to the UI for this reason. Keep
  that habit; it is most of why this project reads as credible.
- Tests: `cd api && .venv/Scripts/python -m pytest -q`.

## Background jobs left running

Kill these when you pick up:

- CPU oil training, writing `api/weights/oil_seg_cpu.pt`.
- CUDA wheel download into `api/.wheels/` — abandon it, use Colab.
