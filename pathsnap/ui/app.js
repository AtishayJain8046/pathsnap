"use strict";

const byId = (id) => document.getElementById(id);
const form = byId("predict-form");
const select = byId("trip-select");
const runButton = byId("predict-button");
const fitButton = byId("fit-button");
const requestStatus = byId("request-status");
const mapState = byId("map-state");
const checkboxes = [...document.querySelectorAll("[data-layer]")];
const number = new Intl.NumberFormat("en-US");
const map = L.map("map", {preferCanvas: true, attributionControl: false}).setView([0, 0], 2);
map.zoomControl.setPosition("topright");
L.control.scale({imperial: false, position: "bottomleft"}).addTo(map);

const colors = {roads: "#8d9a93", gps: "#47769b", classifier: "#a55018", decoded: "#117453"};
const panes = {roads: 390, gps: 400, classifier: 410, decoded: 420};
for (const [name, zIndex] of Object.entries(panes)) {
  map.createPane(name).style.zIndex = zIndex;
}
let layers = {};
let bounds = null;

const coordinates = ([lon, lat]) => [lat, lon];
const percent = (value) => value === null || value === undefined ? "—" : `${(value * 100).toFixed(2)}%`;
const label = (id) => `Trip ${id}`;

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
  const timer = setTimeout(() => controller.abort(), 90000);
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

function setLayer(checkbox) {
  const layer = layers[checkbox.dataset.layer];
  if (!layer) return;
  if (checkbox.checked) layer.addTo(map);
  else map.removeLayer(layer);
}

function fitRoute() {
  if (bounds?.isValid()) map.fitBounds(bounds, {padding: [30, 30], maxZoom: 18});
}

function renderMap(data) {
  for (const layer of Object.values(layers)) map.removeLayer(layer);
  const {gps, roads, classifier, decoded} = data.layers;
  const points = gps.map(coordinates);
  const line = (paths, name, weight, opacity) => L.polyline(
    paths.map((path) => path.map(coordinates)),
    {pane: name, color: colors[name], weight, opacity, interactive: false},
  );
  layers = {
    roads: line(roads, "roads", 2, .45),
    gps: L.layerGroup(points.map((point) => L.circleMarker(point, {
      pane: "gps", radius: 2, color: colors.gps, stroke: false, fillOpacity: .75, interactive: false,
    }))),
    classifier: line(classifier, "classifier", 3, .8),
    decoded: line(decoded, "decoded", 5, .9),
  };
  for (const checkbox of checkboxes) {
    checkbox.disabled = data.layers[checkbox.dataset.layer].length === 0;
    if (!checkbox.disabled) setLayer(checkbox);
  }
  bounds = L.latLngBounds(points);
  for (const path of decoded) for (const point of path) bounds.extend(coordinates(point));
  fitButton.disabled = !bounds.isValid();
  mapState.hidden = bounds.isValid();
  if (!mapState.hidden) mapState.textContent = "This trip has no usable map coordinates.";
  byId("map-detail").textContent = `${label(data.trip_id)} · ${number.format(data.observations)} GPS readings`;
  requestAnimationFrame(() => {
    map.invalidateSize();
    fitRoute();
  });
}

function renderScores(data) {
  const score = data.score;
  byId("decoded-score").textContent = percent(score?.decoded_top1);
  byId("classifier-score").textContent = percent(score?.classifier_top1);
  byId("reachability").textContent = percent(data.directed_reachability);
  byId("matched").textContent = `${number.format(data.matched_observations)} / ${number.format(data.observations)}`;
  byId("segments").textContent = number.format(data.decoded_segments);
  byId("inference").textContent = `${(data.inference_ms / 1000).toFixed(2)} s`;

  const delta = byId("score-delta");
  delta.hidden = score?.decoded_top1 == null || score?.classifier_top1 == null;
  if (!delta.hidden) {
    const gain = 100 * (score.decoded_top1 - score.classifier_top1);
    delta.textContent = `${gain > 0 ? "+" : ""}${gain.toFixed(2)} pp`;
    delta.dataset.sign = gain < 0 ? "negative" : "positive";
  }
  byId("score-detail").textContent = score?.labeled_observations
    ? `${number.format(score.decoded_correct)} correct of ${number.format(score.labeled_observations)} labeled readings`
    : "Reference score unavailable.";
}

async function runTrip() {
  const tripId = select.value;
  if (!tripId || runButton.disabled) return;
  runButton.disabled = select.disabled = true;
  runButton.textContent = "Running…";
  setStatus(`Scoring ${label(tripId).toLowerCase()}…`);
  if (!mapState.hidden) mapState.textContent = `Decoding ${label(tripId).toLowerCase()}…`;
  try {
    const data = await jsonRequest("/predict", {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({trip_id: tripId}),
    });
    renderMap(data);
    renderScores(data);
    setStatus(`${label(data.trip_id)} scored successfully.`);
    setConnection("Server ready", "ready");
  } catch (error) {
    const message = error.name === "AbortError" ? "The request timed out." : error.message;
    setStatus(`${message} Run prediction to retry.`, "error");
    if (!mapState.hidden) mapState.textContent = "Prediction unavailable.";
  } finally {
    runButton.disabled = select.disabled = false;
    runButton.textContent = "Run prediction";
  }
}

async function start() {
  try {
    const {trip_ids: ids} = await jsonRequest("/trips");
    if (!ids.length) throw new Error("No evaluator trips are available.");
    select.replaceChildren(...ids.map((id) => new Option(label(id), id)));
    runButton.disabled = select.disabled = false;
    setConnection("Server ready", "ready");
    await runTrip();
  } catch (error) {
    setStatus(`${error.message} Refresh to retry.`, "error");
    mapState.textContent = "Evaluator data unavailable.";
    setConnection("Server unavailable", "error");
  }
}

form.addEventListener("submit", (event) => { event.preventDefault(); void runTrip(); });
fitButton.addEventListener("click", fitRoute);
for (const checkbox of checkboxes) checkbox.addEventListener("change", () => setLayer(checkbox));
void start();
