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
| `trident/vision` | Not started |
| `trident/mission` | Not started |
| `web/` | Not started |

## Running it

```bash
cd api
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest -q
```

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
