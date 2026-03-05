export const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type FederationStatus = {
  status: "idle" | "ready" | "aggregating" | "complete" | string;
  model_published: boolean;
  current_round: number;
  total_rounds: number;
  published_at: string | null;
  started_at: string | null;
  completed_at: string | null;
  clients: Record<string, ClientInfo>;
};

export type ClientInfo = {
  rounds_submitted: number;
  num_samples: number;
  last_seen: string | null;
  metrics: {
    train_loss?: number;
    eval_loss?: number;
    accuracy?: number;
  };
};

export type Results = {
  rounds: number[];
  accuracy: (number | null)[];
  train_loss: (number | null)[];
  eval_loss: (number | null)[];
};

export type GlobalModelJson = {
  layers: { kernel: number[][]; bias: number[] }[];
};

export type TrainResult = {
  submitted: boolean;
  num_samples: number;
  metrics: { train_loss: number; eval_loss: number; accuracy: number };
};

// ── Server actions ─────────────────────────────────────────────────────────

export async function publishModel() {
  const r = await fetch(`${API}/api/model/publish`, { method: "POST" });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function startAggregation(
  num_rounds: number,
  epochs_per_round: number,
  learning_rate: number
) {
  const r = await fetch(`${API}/api/aggregate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ num_rounds, epochs_per_round, learning_rate }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function resetAll() {
  const r = await fetch(`${API}/api/reset`, { method: "DELETE" });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function fetchStatus(): Promise<FederationStatus> {
  const r = await fetch(`${API}/api/status`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function fetchResults(): Promise<Results> {
  const r = await fetch(`${API}/api/results`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

// ── Hospital (browser-side FL) ────────────────────────────────────────────

/** Global model weights in TF.js format (kernel transposed to [in, out]).
 *  The browser uses these to initialise TF.js, trains locally, then submits only weights.
 *  Patient data NEVER leaves the browser tab. */
export async function fetchGlobalModelJson(): Promise<GlobalModelJson> {
  const r = await fetch(`${API}/api/model/global/json`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

/** Fetch a demo UCI CSV dataset. For real deployments hospitals use their own local files
 *  via the browser FileReader API — the raw data never leaves the machine. */
export async function fetchSampleData(name: string): Promise<string> {
  const r = await fetch(`${API}/api/data/sample/${encodeURIComponent(name)}`);
  if (!r.ok) throw new Error(await r.text());
  return r.text();
}

/** Submit weight updates that were trained locally in the browser.
 *  Only the model deltas travel to the server — never the raw CSV rows. */
export async function submitWeights(
  name: string,
  payload: {
    layers: { kernel: number[][]; bias: number[] }[];
    num_samples: number;
    metrics: { train_loss: number; eval_loss: number; accuracy: number };
  }
): Promise<TrainResult> {
  const r = await fetch(`${API}/api/hospital/${encodeURIComponent(name)}/submit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
