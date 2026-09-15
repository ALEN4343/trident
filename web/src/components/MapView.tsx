import { useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import type { Analysis, Drift, Mission, Telemetry } from "../types";

const STYLE: maplibregl.StyleSpecification = {
  version: 8,
  // Symbol layers refuse to render without a glyph endpoint, and the failure
  // is silent apart from a style error.
  glyphs: "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf",
  sources: {
    // Satellite rather than a street map: the subject is open water, where a
    // road basemap renders as an empty void. Esri World Imagery is keyless,
    // unlike CARTO and Stadia, which now gate their basemaps.
    // Note the {y}/{x} ordering -- this service is row before column.
    base: {
      type: "raster",
      tiles: [
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      ],
      tileSize: 256,
      maxzoom: 19,
      attribution: "Esri, Maxar, Earthstar Geographics",
    },
  },
  layers: [
    { id: "base", type: "raster", source: "base" },
    // Knocks the imagery back so cyan detections and violet routes stay
    // legible over bright coastline and surf.
    {
      id: "base-dim",
      type: "background",
      paint: { "background-color": "#06090e", "background-opacity": 0.28 },
    },
  ],
};

const EMPTY: GeoJSON.FeatureCollection = { type: "FeatureCollection", features: [] };

interface Props {
  analysis: Analysis | null;
  drift: Drift | null;
  reverseDrift: Drift | null;
  mission: Mission | null;
  telemetry: Telemetry | null;
  selected: string | null;
  onSelect: (id: string | null) => void;
}

export default function MapView({
  analysis,
  drift,
  reverseDrift,
  mission,
  telemetry,
  selected,
  onSelect,
}: Props) {
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const vessel = useRef<maplibregl.Marker | null>(null);
  // State, not a ref: the data effects below run before the map finishes
  // loading, and a ref would not re-trigger them once it does.
  const [loaded, setLoaded] = useState(false);
  const select = useRef(onSelect);
  select.current = onSelect;

  useEffect(() => {
    if (!container.current || map.current) return;

    const instance = new maplibregl.Map({
      container: container.current,
      style: STYLE,
      center: [76.2422, 9.9658],
      zoom: 13,
      attributionControl: { compact: true },
    });
    instance.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    instance.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-right");
    map.current = instance;

    instance.on("load", () => {
      for (const id of [
        "footprint",
        "uncertainty",
        "detections",
        "drift-track",
        "drift-cloud",
        "reverse-track",
        "route",
        "waypoints",
        "observed",
        // Not "base": the raster basemap already owns that source id, and the
        // duplicate throws inside the load handler, taking every layer after
        // it with it.
        "home",
      ]) {
        instance.addSource(id, { type: "geojson", data: EMPTY });
      }

      instance.addLayer({
        id: "footprint-fill",
        type: "fill",
        source: "footprint",
        paint: { "fill-color": "#22d3ee", "fill-opacity": 0.07 },
      });
      instance.addLayer({
        id: "footprint-line",
        type: "line",
        source: "footprint",
        paint: { "line-color": "#22d3ee", "line-width": 1, "line-dasharray": [3, 2], "line-opacity": 0.55 },
      });

      instance.addLayer({
        id: "drift-cloud-pts",
        type: "circle",
        source: "drift-cloud",
        paint: { "circle-radius": 2, "circle-color": "#22d3ee", "circle-opacity": 0.28 },
      });
      instance.addLayer({
        id: "drift-track-line",
        type: "line",
        source: "drift-track",
        paint: { "line-color": "#22d3ee", "line-width": 2.5, "line-opacity": 0.9 },
      });
      instance.addLayer({
        id: "reverse-track-line",
        type: "line",
        source: "reverse-track",
        paint: {
          "line-color": "#fbbf24",
          "line-width": 2,
          "line-opacity": 0.8,
          "line-dasharray": [2, 1.5],
        },
      });

      instance.addLayer({
        id: "route-line",
        type: "line",
        source: "route",
        paint: { "line-color": "#a78bfa", "line-width": 2, "line-opacity": 0.85 },
      });
      instance.addLayer({
        id: "observed-pts",
        type: "circle",
        source: "observed",
        paint: {
          "circle-radius": 4,
          "circle-color": "transparent",
          "circle-stroke-color": "#55697e",
          "circle-stroke-width": 1,
        },
      });

      instance.addLayer({
        id: "uncertainty-fill",
        type: "circle",
        source: "uncertainty",
        paint: {
          "circle-radius": ["get", "px"],
          "circle-color": "#22d3ee",
          "circle-opacity": 0.08,
          "circle-stroke-color": "#22d3ee",
          "circle-stroke-width": 0.5,
          "circle-stroke-opacity": 0.3,
        },
      });

      instance.addLayer({
        id: "detection-pts",
        type: "circle",
        source: "detections",
        paint: {
          "circle-radius": ["case", ["get", "selected"], 9, 6],
          "circle-color": ["get", "colour"],
          "circle-stroke-color": "#06090e",
          "circle-stroke-width": 1.5,
        },
      });

      instance.addLayer({
        id: "waypoint-pts",
        type: "circle",
        source: "waypoints",
        paint: {
          "circle-radius": 11,
          "circle-color": ["case", ["get", "signoff"], "#f0514f", "#a78bfa"],
          "circle-opacity": 0.9,
        },
      });
      instance.addLayer({
        id: "waypoint-labels",
        type: "symbol",
        source: "waypoints",
        layout: {
          "text-field": ["get", "seq"],
          "text-size": 12,
          "text-font": ["Open Sans Regular"],
        },
        paint: { "text-color": "#06090e" },
      });

      instance.addLayer({
        id: "home-pt",
        type: "circle",
        source: "home",
        paint: {
          "circle-radius": 7,
          "circle-color": "#34d399",
          "circle-stroke-color": "#06090e",
          "circle-stroke-width": 2,
        },
      });

      instance.on("click", "detection-pts", (event) => {
        const feature = event.features?.[0];
        if (!feature) return;
        select.current(feature.properties?.id ?? null);
        new maplibregl.Popup({ offset: 12 })
          .setLngLat((feature.geometry as GeoJSON.Point).coordinates as [number, number])
          .setHTML(
            `<b>${feature.properties?.label}</b><br/>` +
              `${feature.properties?.conf}% confidence<br/>` +
              `${feature.properties?.coords}<br/>` +
              `<span style="color:#7b8ea3">${feature.properties?.note}</span>`,
          )
          .addTo(instance);
      });
      instance.on("mouseenter", "detection-pts", () => {
        instance.getCanvas().style.cursor = "pointer";
      });
      instance.on("mouseleave", "detection-pts", () => {
        instance.getCanvas().style.cursor = "";
      });

      setLoaded(true);
    });

    return () => {
      instance.remove();
      map.current = null;
      setLoaded(false);
    };
  }, []);

  // Detections, footprint and uncertainty
  useEffect(() => {
    const instance = map.current;
    if (!instance || !loaded || !analysis) return;

    const set = (id: string, data: GeoJSON.FeatureCollection) =>
      (instance.getSource(id) as maplibregl.GeoJSONSource | undefined)?.setData(data);

    const located = analysis.items.filter((i) => i.lat !== null && i.lon !== null);

    set("detections", {
      type: "FeatureCollection",
      features: located.map((item) => ({
        type: "Feature",
        geometry: { type: "Point", coordinates: [item.lon!, item.lat!] },
        properties: {
          id: item.id,
          label: item.label,
          conf: (item.confidence * 100).toFixed(0),
          coords: `${item.lat!.toFixed(5)}, ${item.lon!.toFixed(5)}`,
          note: item.auto_collect ? "autonomous collection permitted" : "requires human sign-off",
          colour: item.auto_collect ? "#22d3ee" : "#f0514f",
          selected: item.id === selected,
        },
      })),
    });

    // Uncertainty is drawn in real metres, so the circle shrinks as you zoom
    // out. A fixed-pixel halo would imply a precision the fix does not have.
    const metresToPixels = (metres: number, lat: number) => {
      const zoom = instance.getZoom();
      const resolution = (156543.03392 * Math.cos((lat * Math.PI) / 180)) / 2 ** zoom;
      return Math.max(3, metres / resolution);
    };

    set("uncertainty", {
      type: "FeatureCollection",
      features: located
        .filter((i) => i.uncertainty_m)
        .map((item) => ({
          type: "Feature",
          geometry: { type: "Point", coordinates: [item.lon!, item.lat!] },
          properties: { px: metresToPixels(item.uncertainty_m!, item.lat!) },
        })),
    });

    set(
      "footprint",
      analysis.footprint
        ? {
            type: "FeatureCollection",
            features: [
              {
                type: "Feature",
                geometry: {
                  type: "Polygon",
                  coordinates: [analysis.footprint.map(([lat, lon]) => [lon, lat])],
                },
                properties: {},
              },
            ],
          }
        : EMPTY,
    );

    if (analysis.capture.pose) {
      const { lat, lon } = analysis.capture.pose;
      instance.easeTo({ center: [lon, lat], zoom: Math.max(instance.getZoom(), 16), duration: 700 });
    }
  }, [analysis, selected, loaded]);

  // Drift layers
  useEffect(() => {
    const instance = map.current;
    if (!instance || !loaded) return;
    const set = (id: string, data: GeoJSON.FeatureCollection) =>
      (instance.getSource(id) as maplibregl.GeoJSONSource | undefined)?.setData(data);

    if (drift) {
      set("drift-track", {
        type: "FeatureCollection",
        features: [
          {
            type: "Feature",
            geometry: {
              type: "LineString",
              coordinates: drift.track.map((p) => [p.lon, p.lat]),
            },
            properties: {},
          },
        ],
      });
      const cloud = drift.geojson.features.filter(
        (f) => f.geometry.type === "MultiPoint",
      ) as GeoJSON.Feature[];
      set("drift-cloud", { type: "FeatureCollection", features: cloud });
    } else {
      set("drift-track", EMPTY);
      set("drift-cloud", EMPTY);
    }

    set(
      "reverse-track",
      reverseDrift
        ? {
            type: "FeatureCollection",
            features: [
              {
                type: "Feature",
                geometry: {
                  type: "LineString",
                  coordinates: reverseDrift.track.map((p) => [p.lon, p.lat]),
                },
                properties: {},
              },
            ],
          }
        : EMPTY,
    );
  }, [drift, reverseDrift, loaded]);

  // Mission route
  useEffect(() => {
    const instance = map.current;
    if (!instance || !loaded) return;
    const set = (id: string, data: GeoJSON.FeatureCollection) =>
      (instance.getSource(id) as maplibregl.GeoJSONSource | undefined)?.setData(data);

    if (!mission) {
      set("route", EMPTY);
      set("waypoints", EMPTY);
      set("observed", EMPTY);
      set("home", EMPTY);
      return;
    }

    const path: [number, number][] = [
      [mission.base.lon, mission.base.lat],
      ...mission.waypoints.map((w) => [w.lon, w.lat] as [number, number]),
      [mission.base.lon, mission.base.lat],
    ];

    set("route", {
      type: "FeatureCollection",
      features: [{ type: "Feature", geometry: { type: "LineString", coordinates: path }, properties: {} }],
    });

    set("waypoints", {
      type: "FeatureCollection",
      features: mission.waypoints.map((w) => ({
        type: "Feature",
        geometry: { type: "Point", coordinates: [w.lon, w.lat] },
        properties: { seq: String(w.seq), signoff: w.needs_signoff },
      })),
    });

    // The observed positions, so the offset between "where we saw it" and
    // "where we are going" is visible rather than asserted.
    set("observed", {
      type: "FeatureCollection",
      features: mission.waypoints.map((w) => ({
        type: "Feature",
        geometry: { type: "Point", coordinates: [w.observed_lon, w.observed_lat] },
        properties: {},
      })),
    });

    set("home", {
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          geometry: { type: "Point", coordinates: [mission.base.lon, mission.base.lat] },
          properties: {},
        },
      ],
    });

    const bounds = new maplibregl.LngLatBounds();
    path.forEach((c) => bounds.extend(c));
    instance.fitBounds(bounds, { padding: 90, duration: 800, maxZoom: 17 });
  }, [mission, loaded]);

  // Live vessel
  useEffect(() => {
    const instance = map.current;
    if (!instance || !loaded) return;

    if (!telemetry) {
      vessel.current?.remove();
      vessel.current = null;
      return;
    }

    if (!vessel.current) {
      const el = document.createElement("div");
      el.style.cssText =
        "width:0;height:0;border-left:8px solid transparent;border-right:8px solid transparent;" +
        "border-bottom:18px solid #22d3ee;filter:drop-shadow(0 0 7px #22d3ee);";
      vessel.current = new maplibregl.Marker({ element: el, rotationAlignment: "map" })
        .setLngLat([telemetry.lon, telemetry.lat])
        .addTo(instance);
    }
    vessel.current.setLngLat([telemetry.lon, telemetry.lat]);
    vessel.current.setRotation(telemetry.heading_deg);
  }, [telemetry, loaded]);

  return <div className="map" ref={container} />;
}
