"use client";

import {useCallback, useEffect, useRef, useState} from "react";
import {FIT_ROUTE_MESSAGE, jsonRequest, number, percent, predictTrip, tripLabel} from "../../pathsnap/ui/shared.mjs";

export default function Evaluator() {
  const [trips, setTrips] = useState([]);
  const [selected, setSelected] = useState("");
  const [connection, setConnection] = useState("waiting");
  const [busy, setBusy] = useState(true);
  const [status, setStatus] = useState("Loading available trips…");
  const [failed, setFailed] = useState(false);
  const [result, setResult] = useState(null);
  const [mapReady, setMapReady] = useState(false);
  const frame = useRef(null);

  const run = useCallback(async (tripId) => {
    if (!tripId) return;
    setBusy(true);
    setFailed(false);
    setStatus(`Scoring ${tripLabel(tripId).toLowerCase()}…`);
    try {
      const data = await predictTrip("/api", tripId);
      setMapReady(false);
      setResult(data);
      setStatus(`${tripLabel(data.trip_id)} scored successfully.`);
    } catch (error) {
      const message = error.name === "AbortError" ? "The request timed out." : error.message;
      setFailed(true);
      setStatus(`${message} Run prediction to retry.`);
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    let active = true;
    jsonRequest("/api/trips").then(({trip_ids: ids}) => {
      if (!active) return;
      if (!ids.length) throw new Error("No evaluator trips are available.");
      setTrips(ids);
      setSelected(ids[0]);
      setConnection("ready");
      void run(ids[0]);
    }).catch((error) => {
      if (!active) return;
      setConnection("error");
      setFailed(true);
      setStatus(`${error.message} Refresh to retry.`);
      setBusy(false);
    });
    return () => { active = false; };
  }, [run]);

  const score = result?.score;
  const gain = score?.decoded_top1 == null || score?.classifier_top1 == null
    ? null : 100 * (score.decoded_top1 - score.classifier_top1);
  const connectionText = {waiting: "Connecting", ready: "Server ready", error: "Server unavailable"};

  return (
    <>
      <header className="site-header">
        <div className="frame header-inner">
          <div className="brand"><span className="brand-mark" aria-hidden="true" />PathSnap</div>
          <span className="connection" data-state={connection}><i />{connectionText[connection]}</span>
        </div>
      </header>
      <main className="frame">
        <div className="page-title">
          <div><h1>Route evaluator</h1><p>Run a trip and compare the classifier with the decoded route.</p></div>
          <a className="api-link" href="/api/openapi.json" target="_blank" rel="noopener">API schema <span aria-hidden="true">↗</span></a>
        </div>
        <div className="workspace">
          <section className="panel controls-panel" aria-labelledby="controls-title">
            <h2 id="controls-title">Trip selection</h2>
            <form onSubmit={(event) => { event.preventDefault(); if (!busy) void run(selected); }}>
              <label htmlFor="trip-select">Evaluator trip</label>
              <select id="trip-select" value={selected} disabled={busy || !trips.length}
                onChange={(event) => setSelected(event.target.value)}>
                {!trips.length && <option value="">Loading trips</option>}
                {trips.map((id) => <option key={id} value={id}>{tripLabel(id)}</option>)}
              </select>
              <button className="primary" type="submit" disabled={busy || !selected}>
                {busy ? "Running…" : "Run prediction"}
              </button>
            </form>
            <p className="request-status" role="status" data-state={failed ? "error" : "ready"}>{status}</p>
          </section>
          <section className="panel map-panel" aria-labelledby="map-title">
            <div className="map-heading">
              <div>
                <h2 id="map-title">Folium route map</h2>
                <p id="map-detail">{result
                  ? `${tripLabel(result.trip_id)} · ${number.format(result.observations)} GPS readings`
                  : "Waiting for a trip"}</p>
              </div>
              <button className="secondary" type="button" disabled={!mapReady}
                onClick={() => frame.current?.contentWindow?.postMessage(FIT_ROUTE_MESSAGE, "*")}>Fit route</button>
            </div>
            <div className="map-stage">
              {result
                ? <iframe id="route-map" ref={frame} title="Interactive Folium route map"
                    src={result.map_url} onLoad={() => setMapReady(true)} />
                : <div className="map-state">{failed ? "Prediction unavailable." : "Waiting for the first prediction…"}</div>}
            </div>
          </section>
          <section className="panel score-panel" aria-labelledby="score-title">
            <h2 id="score-title">Evaluation</h2>
            <div className="headline-score">
              <p>Decoded Top-1</p>
              <div className="score-line">
                <strong id="decoded-score">{percent(score?.decoded_top1)}</strong>
                {gain != null && <span className="delta" data-sign={gain < 0 ? "negative" : "positive"}>
                  {gain > 0 ? "+" : ""}{gain.toFixed(2)} pp
                </span>}
              </div>
              <p className="score-detail">{score?.labeled_observations
                ? `${number.format(score.decoded_correct)} correct of ${number.format(score.labeled_observations)} labeled readings`
                : result ? "Reference score unavailable." : "Accuracy appears after prediction."}</p>
            </div>
            <dl className="score-table">
              <div><dt>Classifier Top-1</dt><dd>{percent(score?.classifier_top1)}</dd></div>
              <div><dt>Directed reachability</dt><dd>{percent(result?.directed_reachability)}</dd></div>
              <div><dt>Matched readings</dt><dd>{result
                ? `${number.format(result.matched_observations)} / ${number.format(result.observations)}` : "—"}</dd></div>
              <div><dt>Decoded segments</dt><dd>{result ? number.format(result.decoded_segments) : "—"}</dd></div>
              <div><dt>Inference time</dt><dd>{result ? `${(result.inference_ms / 1000).toFixed(2)} s` : "—"}</dd></div>
            </dl>
          </section>
        </div>
      </main>
    </>
  );
}
