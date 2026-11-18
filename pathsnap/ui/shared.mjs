export const number = new Intl.NumberFormat("en-US");
export const percent = (value) => value == null ? "—" : `${(value * 100).toFixed(2)}%`;
export const tripLabel = (id) => `Trip ${id}`;
export const FIT_ROUTE_MESSAGE = "pathsnap-fit-route";

export async function jsonRequest(path, options = {}) {
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

export async function predictTrip(prefix, tripId) {
  const prediction = await jsonRequest(`${prefix}/predict?include_map=true&map_format=url`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({trip_id: tripId}),
  });
  if (!prediction.map_url?.startsWith("/maps/")) throw new Error("The server did not return a Folium map.");
  return {...prediction, map_url: prefix + prediction.map_url};
}
