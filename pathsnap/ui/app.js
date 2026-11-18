import {FIT_ROUTE_MESSAGE, jsonRequest, number, percent, predictTrip, tripLabel as label} from "./shared.mjs";

const byId = (id) => document.getElementById(id);
const form = byId("predict-form");
const select = byId("trip-select");
const runButton = byId("predict-button");
const fitButton = byId("fit-button");
const requestStatus = byId("request-status");
const mapState = byId("map-state");
const mapFrame = byId("route-map");

function setConnection(text, state) {
  byId("connection-text").textContent = text;
  byId("connection").dataset.state = state;
}

function setStatus(text, state = "ready") {
  requestStatus.textContent = text;
  requestStatus.dataset.state = state;
}

function renderMap(data) {
  fitButton.disabled = true;
  mapFrame.hidden = false;
  mapFrame.src = data.map_url;
  mapState.hidden = true;
  byId("map-detail").textContent = `${label(data.trip_id)} · ${number.format(data.observations)} GPS readings`;
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
    const data = await predictTrip("", tripId);
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
mapFrame.addEventListener("load", () => { fitButton.disabled = !mapFrame.hasAttribute("src"); });
fitButton.addEventListener("click", () => mapFrame.contentWindow?.postMessage(FIT_ROUTE_MESSAGE, "*"));
void start();
