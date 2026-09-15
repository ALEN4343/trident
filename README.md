# TRIDENT

Marine pollution intelligence for SDG 14.

Most pollution tools stop at "this water is dirty." TRIDENT closes the loop:

```
DETECT  ->  LOCALISE  ->  ATTRIBUTE  ->  FORECAST  ->  DISPATCH  ->  VERIFY
  what       exact         where it      where it     cleanup      before /
  and how    lat/lon per   came from     will be      route with   after
  much       object        (hindcast)    in 6 h       human gate   severity
```

## Why each stage exists

**Detect.** Three modalities, because pollution does not share one visual
signature. Countable objects get instance segmentation. Oil sheen and algal
bloom have no edges, so they get semantic segmentation driven by spectral
indices. And a third class exists purely to be *rejected*: sea foam, sun
glint, vessel wakes and Sargassum are mistaken for pollution far more often
than pollution is missed. The report states what was considered and thrown
out.

**Localise.** Camera pose plus a pinhole ray/plane intersection gives every
detection its own coordinate, not one pin per photo. The same projection
yields ground sample distance, so object sizes are in centimetres and the
imaged water area is in square metres, which is what makes a density figure
mean anything.

**Attribute and forecast.** Debris position plus live ocean current and wind
drives a Lagrangian particle model. Forward, it predicts where a cleanup crew
should actually go, since a boat arriving in four hours meets water that has
moved kilometres. Backward, the same integrator hindcasts where the debris
entered the water.

**Dispatch.** Detections cluster into hotspots, a capacity- and
battery-constrained route is planned against *predicted* positions, and a
human approves the mission and every uncertain or hazardous pickup.

## Status

| Module | State |
| --- | --- |
| `trident/taxonomy` | Done - 20 pollution classes, 7 confusers, physics and policy per class |
| `trident/geo` | Done - pose model, ray/plane projection, footprint, uncertainty |
| `trident/drift` | Done - live Open-Meteo forcing, RK4 advection, forward and reverse |
| `trident/severity` | Done - five-component MPSI with a confidence band |
| `trident/vision` | Done - YOLO instance branch, physics branch, confuser rejection |
| `trident/sar` | Done - trained DeepLabv3+ oil segmenter, carried over from SAMUDRA |
| `trident/mission` | Done - clustering, drift-aware routing, approval gates, simulator |
| `trident/api` | Done - upload, drift, planning, gates, telemetry socket |
| `web/` | Done - mission control UI, optical and SAR modes |

The end-to-end loop runs: upload an image, get per-object coordinates on a
satellite map, forecast drift, plan a route, clear the approval gates, and
watch the simulated vessel fly it.

## Running it

Two processes. Backend:

```bash
cd api && python -m venv .venv && .venv/Scripts/python -m pip install -e ".[dev,vision]" && .venv/Scripts/python -m uvicorn trident.api.main:app --port 8000
```

Frontend:

```bash
cd web && npm install && npm run dev
```

Then open http://localhost:5173. Tests:

```bash
cd api && .venv/Scripts/python -m pytest -q
```

## Two sensors, not two opinions

Oil is detected twice over, by genuinely different physics, and the two are
deliberately not merged:

**Optical** (`trident/vision`) reads a thin interference film in visible light.
Oil is iridescent at low saturation, and it damps capillary waves, so a slick
is a smooth anomaly inside an otherwise wave-textured field — the optical
equivalent of the dark-spot step used on radar. Costs nothing to run and works
on any photograph.

**SAR** (`trident/sar`) is a trained DeepLabv3+ over Sentinel-1 and PALSAR
backscatter, carried over from SAMUDRA. Radar does not see colour at all; it
sees that oil has flattened the sea surface so the return drops. It works at
night, through cloud, and over whole coastlines.

They are exposed as separate modes because handing a phone photograph to a
model trained on radar backscatter produces a confident mask of nothing. The
per-sensor evaluation in training exists for the same reason: PALSAR is L-band
and Sentinel-1 is C-band, so a model scoring well on one and badly on the other
has learned a sensor rather than oil.

### Training the oil segmenter

```bash
cd api && .venv/Scripts/python scripts/train_oil.py --epochs 12 --batch-size 8
```

Points at the SOS dataset (6,455 train / 1,615 test image-mask pairs) plus
5,538 image-level labels driving an auxiliary head. The checkpoint records
`representative`, true only for a full GPU run, and that flag is surfaced in
the UI next to every mask it produces.

## Known limits

Honest about what this is at the end of a short build:

- **The instance detector is COCO-pretrained, not fine-tuned.** It reliably
  finds bottles, cups and glassware at close range and misses small litter in
  wide aerial shots, which is most of a real survey frame. Fine-tuning on
  merged TACO and TrashCan against the taxonomy mapping is the fix, and the
  mapping table already exists for it.
- **The physics branch is calibrated on synthetic scenes.** The oil / foam /
  glint separation behaves correctly on constructed fixtures and on some real
  photographs, but the thresholds have not been fitted to a labelled set.
- **Water segmentation is a heuristic**, not a learned segmenter. It handles
  sky, vegetation and dry sand; it will mislabel very turbid brown water.
- Sessions live in memory, so a backend restart clears them.

## Design notes

**The taxonomy is the spine.** `trident/taxonomy/classes.yaml` is the only
place class behaviour is defined. Severity weights, drift windage, and whether
a vehicle may collect an item without human sign-off all read from it. There
are no hardcoded class lists elsewhere, so retuning the system is a data edit.

**Windage comes from the taxonomy.** A styrofoam block rides high and takes
4.5% of the wind; a waterlogged net takes 0.5%. Released from the same pixel
they separate measurably over a six hour forecast, because the drift model
asks the taxonomy how each item floats.

**Nothing is hardcoded to "HIGH".** The severity index is five weighted,
individually inspectable components, reported with a confidence band. A wide
band means the imagery is the problem, not the water.

**Offline is a first-class path.** Venue networks fail. When the forcing API
is unreachable the drift model falls back to a deterministic synthetic field
and every result derived from it is labelled synthetic rather than passed off
as a forecast.

## Data and prior art

Datasets: [TACO](http://tacodataset.org/),
[MARIDA](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0262247),
TrashCan 1.0, ICRA'19 Trash.

Forcing: [Open-Meteo Marine](https://open-meteo.com/en/docs/marine-weather-api)
and Weather APIs, no key required.

Methods: Floating Debris Index (Biermann et al. 2020) for spectral debris
detection; GLCM texture for oil slick / look-alike separation;
[OpenDrift](https://opendrift.github.io/) as the reference Lagrangian
implementation; [SeaClear](https://seaclear-project.eu/) as the reference
multi-robot cleanup architecture.
