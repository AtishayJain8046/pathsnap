"use strict";

const byId = (id) => document.getElementById(id);
const form = byId("predict-form");
const select = byId("trip-select");
const runButton = byId("predict-button");
const fitButton = byId("fit-button");
const requestStatus = byId("request-status");
const mapState = byId("map-state");
const mapNote = byId("map-note");
const layerToggles = [...document.querySelectorAll("[data-layer]")];
const number = new Intl.NumberFormat("en-US");
const colors = {
  gps: "#315fdb",
  roads: "#76817b",
  classifier: "#e2703a",
  decoded: "#087b5b",
};

let map = null;
let bounds = null;
let dataLayers = {};

const percent = (value) => value === null || value === undefined ? "—" : `${(value * 100).toFixed(2)}%`;
const tripLabel = (id) => `Trip ${id}`;

function setConnection(text, state) {
  byId("connection-text").textContent = text;
  byId("connection").dataset.state = state;
}

function setStatus(text, state = "ready") {
  requestStatus.textContent = text;
  requestStatus.dataset.state = state;
}

async function jsonRequest(path, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 120000);
  try {
    const response = await fetch(path, {...options, signal: controller.signal});
    if (!response.ok) {
      const payload = await response.json().catch(() => null);
      const detail = typeof payload?.detail === "string" ? payload.detail : `Request failed (${response.status}).`;
      throw new Error(detail);
    }
    return await response.json();
  } finally {
    clearTimeout(timer);
  }
}

function loadGoogleMaps(apiKey, mapId) {
  return new Promise((resolve, reject) => {
    const callbackName = `initPathSnapMap_${Date.now()}`;
    const timeout = setTimeout(() => {
      delete window[callbackName];
      reject(new Error("Google Maps took too long to load."));
    }, 20000);

    window.gm_authFailure = () => reject(new Error("Google Maps rejected the configured API key."));
    window[callbackName] = () => {
      clearTimeout(timeout);
      delete window[callbackName];
      const options = {
        center: {lat: 0, lng: 0},
        zoom: 2,
        mapTypeId: "roadmap",
        clickableIcons: false,
        fullscreenControl: true,
        mapTypeControl: true,
        streetViewControl: true,
        gestureHandling: "greedy",
      };
      if (mapId) options.mapId = mapId;
      map = new google.maps.Map(byId("map"), options);
      resolve(map);
    };

    const script = document.createElement("script");
    const params = new URLSearchParams({
      key: apiKey,
      loading: "async",
      v: "weekly",
      callback: callbackName,
    });
    script.src = `https://maps.googleapis.com/maps/api/js?${params}`;
    script.async = true;
    script.onerror = () => {
      clearTimeout(timeout);
      delete window[callbackName];
      reject(new Error("Google Maps could not be downloaded."));
    };
    document.head.append(script);
  });
}

function geoJson(paths, type) {
  return {type: "Feature", geometry: {type, coordinates: paths}, properties: {}};
}

function createLayer(geoJsonData, style) {
  const layer = new google.maps.Data({map: null});
  layer.addGeoJson(geoJsonData);
  layer.setStyle(style);
  return layer;
}

function setLayer(toggle) {
  const layer = dataLayers[toggle.dataset.layer];
  if (layer) layer.setMap(toggle.checked ? map : null);
}

function fitRoute() {
  if (!map || !bounds || bounds.isEmpty()) return;
  map.fitBounds(bounds, 54);
  google.maps.event.addListenerOnce(map, "idle", () => {
    if (map.getZoom() > 18) map.setZoom(18);
  });
}

function renderMap(data) {
  if (!map) return;
  for (const layer of Object.values(dataLayers)) layer.setMap(null);

  const {gps, roads, classifier, decoded} = data.layers;
  dataLayers = {
    roads: createLayer(geoJson(roads, "MultiLineString"), {
      strokeColor: colors.roads,
      strokeOpacity: 0.48,
      strokeWeight: 2,
      zIndex: 1,
    }),
    gps: createLayer(geoJson(gps, "MultiPoint"), {
      icon: {
        path: google.maps.SymbolPath.CIRCLE,
        scale: 3.6,
        fillColor: colors.gps,
        fillOpacity: 0.92,
        strokeColor: "#ffffff",
        strokeOpacity: 0.9,
        strokeWeight: 1,
      },
      zIndex: 4,
    }),
    classifier: createLayer(geoJson(classifier, "MultiLineString"), {
      strokeColor: colors.classifier,
      strokeOpacity: 0.82,
      strokeWeight: 4,
      zIndex: 2,
    }),
    decoded: createLayer(geoJson(decoded, "MultiLineString"), {
      strokeColor: colors.decoded,
      strokeOpacity: 0.95,
      strokeWeight: 6,
      zIndex: 3,
    }),
  };

  for (const toggle of layerToggles) {
    toggle.disabled = data.layers[toggle.dataset.layer].length === 0;
    setLayer(toggle);
  }

  bounds = new google.maps.LatLngBounds();
  const extend = ([lng, lat]) => bounds.extend({lat, lng});
  gps.forEach(extend);
  decoded.forEach((path) => path.forEach(extend));
  fitButton.disabled = bounds.isEmpty();
  mapState.hidden = !bounds.isEmpty();
  if (!mapState.hidden) mapState.textContent = "This trip has no usable map coordinates.";
  byId("map-detail").textContent = `${tripLabel(data.trip_id)} · ${number.format(data.observations)} GPS readings`;
  fitRoute();
}

function renderScores(data) {
  const score = data.score;
  byId("decoded-score").textContent = percent(score?.decoded_top1);
  byId("classifier-score").textContent = percent(score?.classifier_top1);
  byId("reachability").textContent = percent(data.directed_reachability);
  byId("matched").textContent = `${number.format(data.matched_observations)} / ${number.format(data.observations)}`;
  byId("segments").textContent = number.format(data.decoded_segments);
  byId("inference").textContent = data.inference_ms < 1000
    ? `${number.format(Math.round(data.inference_ms))} ms`
    : `${(data.inference_ms / 1000).toFixed(2)} s`;
  byId("observations").textContent = number.format(data.observations);
  byId("labeled").textContent = score ? number.format(score.labeled_observations) : "—";
  byId("classifier-correct").textContent = score ? number.format(score.classifier_correct) : "—";
  byId("decoder-correct").textContent = score ? number.format(score.decoded_correct) : "—";

  const delta = byId("score-delta");
  delta.hidden = score?.decoded_top1 == null || score?.classifier_top1 == null;
  if (!delta.hidden) {
    const gain = 100 * (score.decoded_top1 - score.classifier_top1);
    delta.textContent = `${gain > 0 ? "+" : ""}${gain.toFixed(2)} pp`;
    delta.dataset.sign = gain < 0 ? "negative" : "positive";
  }
  byId("score-detail").textContent = score?.labeled_observations
    ? `${number.format(score.decoded_correct)} correct of ${number.format(score.labeled_observations)} labeled readings`
    : "Reference score unavailable for this trip.";
}

async function runTrip() {
  const tripId = select.value;
  if (!tripId || runButton.disabled) return;
  runButton.disabled = true;
  select.disabled = true;
  runButton.querySelector("span").textContent = "Evaluating…";
  setStatus(`Running classifier and HMM for ${tripLabel(tripId).toLowerCase()}…`);
  if (!mapState.hidden) mapState.textContent = `Decoding ${tripLabel(tripId).toLowerCase()}…`;

  try {
    const data = await jsonRequest("/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({trip_id: tripId}),
    });
    renderScores(data);
    renderMap(data);
    setStatus(`${tripLabel(data.trip_id)} evaluated successfully.`);
    setConnection("Server ready", "ready");
  } catch (error) {
    const message = error.name === "AbortError" ? "The evaluation request timed out." : error.message;
    setStatus(`${message} Run evaluation to retry.`, "error");
    if (!mapState.hidden) mapState.textContent = "Evaluation unavailable.";
  } finally {
    runButton.disabled = false;
    select.disabled = false;
    runButton.querySelector("span").textContent = "Run evaluation";
  }
}

async function configureMap() {
  try {
    const config = await jsonRequest("/config");
    if (!config.google_maps_api_key) {
      mapState.textContent = "Add GOOGLE_MAPS_API_KEY to display the Google basemap.";
      mapNote.textContent = "Evaluation metrics remain available. Configure a Google Maps browser key to draw the route layers.";
      return;
    }
    await loadGoogleMaps(config.google_maps_api_key, config.google_maps_map_id);
    mapState.textContent = "Waiting for the first route…";
  } catch (error) {
    mapState.textContent = error.message;
    mapNote.textContent = "Check the API key, referrer restrictions and Maps JavaScript API status in Google Cloud.";
  }
}

async function start() {
  const mapPromise = configureMap();
  try {
    const {trip_ids: ids} = await jsonRequest("/trips");
    if (!ids.length) throw new Error("No evaluation trips are available.");
    select.replaceChildren(...ids.map((id) => new Option(tripLabel(id), id)));
    runButton.disabled = false;
    select.disabled = false;
    setConnection("Server ready", "ready");
    await mapPromise;
    await runTrip();
  } catch (error) {
    setStatus(`${error.message} Refresh to retry.`, "error");
    if (!map) mapState.textContent = "Evaluation data unavailable.";
    setConnection("Server unavailable", "error");
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  void runTrip();
});
fitButton.addEventListener("click", fitRoute);
for (const toggle of layerToggles) toggle.addEventListener("change", () => setLayer(toggle));

void start();
