import {fileURLToPath} from "node:url";

const api = new URL(process.env.PATHSNAP_API_URL ?? "http://127.0.0.1:8000").origin;

export default {
  turbopack: {root: fileURLToPath(new URL("..", import.meta.url))},
  outputFileTracingRoot: fileURLToPath(new URL("..", import.meta.url)),
  async rewrites() {
    return ["/trips", "/predict", "/health", "/openapi.json", "/maps/:trip_id"].map((path) => ({
      source: `/api${path}`,
      destination: `${api}${path}`,
    }));
  },
};
