"use client";
import { use, useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  fetchStatus, fetchGlobalModelJson, fetchSampleData, submitWeights,
  type FederationStatus, type TrainResult,
} from "@/lib/api";

type Props = { params: Promise<{ name: string }> };

const SAMPLE_DATASETS = [
  { id: "cleveland",   label: "Cleveland (UCI, 303 patients)" },
  { id: "hungarian",   label: "Hungarian (UCI, 294 patients)" },
  { id: "va",          label: "VA Long Beach (UCI, 200 patients)" },
  { id: "switzerland", label: "Switzerland (UCI, 123 patients)" },
];

// ── Data preprocessing (pure browser, no data leaves) ────────────────────────

function parseUCI(text: string): { X: number[][]; y: number[] } {
  const lines = text.trim().split("\n").filter(l => l.trim());
  // Skip header row if present (first cell can't be parsed as a number)
  const start = isNaN(Number(lines[0].split(",")[0].trim())) ? 1 : 0;

  const colMeans = Array(13).fill(0);
  const colCounts = Array(13).fill(0);
  const rows = lines.slice(start).map(line =>
    line.split(",").map(v => (v.trim() === "?" || v.trim() === "" ? NaN : parseFloat(v.trim())))
  ).filter(r => r.length >= 14);

  // Compute column means for NaN imputation
  rows.forEach(r => {
    for (let j = 0; j < 13; j++) {
      if (!isNaN(r[j])) { colMeans[j] += r[j]; colCounts[j]++; }
    }
  });
  colMeans.forEach((_, j) => { colMeans[j] /= colCounts[j] || 1; });

  const X = rows.map(r => r.slice(0, 13).map((v, j) => isNaN(v) ? colMeans[j] : v));
  const y = rows.map(r => r[13] > 0 ? 1 : 0);
  return { X, y };
}

function standardScale(X: number[][]): number[][] {
  const n = X.length, d = X[0].length;
  const mu = Array.from({ length: d }, (_, j) => X.reduce((s, r) => s + r[j], 0) / n);
  const sd = Array.from({ length: d }, (_, j) =>
    Math.sqrt(X.reduce((s, r) => s + (r[j] - mu[j]) ** 2, 0) / n) || 1
  );
  return X.map(r => r.map((v, j) => (v - mu[j]) / sd[j]));
}

function splitTrainTest(X: number[][], y: number[], testRatio = 0.2) {
  const idx = Array.from({ length: X.length }, (_, i) => i);
  // Deterministic shuffle (LCG seed 42)
  let s = 42;
  for (let i = idx.length - 1; i > 0; i--) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    const j = s % (i + 1);
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  const cut = Math.floor(X.length * testRatio);
  const tr = idx.slice(cut), te = idx.slice(0, cut);
  return { xTrain: tr.map(i => X[i]), yTrain: tr.map(i => y[i]),
           xTest: te.map(i => X[i]),  yTest: te.map(i => y[i]), numTrain: tr.length };
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function HospitalMonitor({ params }: Props) {
  const { name } = use(params);
  const displayName = decodeURIComponent(name);
  const capitalName = displayName.charAt(0).toUpperCase() + displayName.slice(1);

  const [status, setStatus]       = useState<FederationStatus | null>(null);
  const [dataMode, setDataMode]   = useState<"sample" | "upload">("sample");
  const [dataset, setDataset]     = useState("cleveland");
  const [file, setFile]           = useState<File | null>(null);
  const [epochs, setEpochs]       = useState(5);
  const [lr, setLr]               = useState(0.01);
  const [training, setTraining]   = useState(false);
  const [progress, setProgress]   = useState<{ epoch: number; total: number; loss: number } | null>(null);
  const [result, setResult]       = useState<TrainResult | null>(null);
  const [error, setError]         = useState<string | null>(null);
  const fileRef                   = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const poll = async () => setStatus(await fetchStatus());
    poll();
    const id = setInterval(poll, 2000);
    return () => clearInterval(id);
  }, []);

  const modelPublished = status?.model_published ?? false;
  const client         = status?.clients?.[displayName];
  const isComplete     = status?.status === "complete";
  const hasData        = dataMode === "sample" || file !== null;

  // ── In-browser training (TF.js) ─────────────────────────────────────────
  const handleTrain = async () => {
    setTraining(true);
    setError(null);
    setProgress(null);

    try {
      // Lazy-load TF.js — only downloaded when user clicks Train
      const tf = await import("@tensorflow/tfjs");

      // 1. Fetch current global model weights (JSON, no .pt binary needed)
      const globalModel = await fetchGlobalModelJson();

      // 2. Load CSV data — entirely in the browser, never sent to server
      let csvText: string;
      if (dataMode === "upload" && file) {
        csvText = await file.text();   // FileReader: data stays in browser tab
      } else {
        csvText = await fetchSampleData(dataset);   // Public demo data from server
      }

      // 3. Preprocess in browser
      const { X, y } = parseUCI(csvText);
      const Xscaled = standardScale(X);
      const { xTrain, yTrain, xTest, yTest, numTrain } = splitTrainTest(Xscaled, y);

      // 4. Build the same architecture as HeartDiseaseNet (13→64→32→1, sigmoid)
      const model = tf.sequential({
        layers: [
          tf.layers.dense({ inputShape: [13], units: 64, activation: "relu" }),
          tf.layers.dense({ units: 32, activation: "relu" }),
          tf.layers.dense({ units: 1, activation: "sigmoid" }),
        ],
      });

      // 5. Load global weights (kernel already in [in, out] format from server)
      const initWeights = globalModel.layers.flatMap(l => [tf.tensor2d(l.kernel), tf.tensor1d(l.bias)]);
      model.setWeights(initWeights);
      // setWeights copies data into model's internal vars — initWeights can now be freed
      initWeights.forEach(t => t.dispose());

      // 6. Compile
      model.compile({ optimizer: tf.train.adam(lr), loss: "binaryCrossentropy", metrics: ["accuracy"] });

      // 7. Train locally — raw data never leaves this browser tab
      const xTrT = tf.tensor2d(xTrain), yTrT = tf.tensor2d(yTrain.map(v => [v]));
      const xTeT = tf.tensor2d(xTest),  yTeT = tf.tensor2d(yTest.map(v => [v]));

      let lastTrainLoss = 0;
      await model.fit(xTrT, yTrT, {
        epochs,
        batchSize: 32,
        callbacks: {
          onEpochEnd: (epoch: number, logs?: Record<string, number>) => {
            lastTrainLoss = logs?.loss ?? 0;
            setProgress({ epoch: epoch + 1, total: epochs, loss: lastTrainLoss });
          },
        },
      });

      // 8. Evaluate on held-out test set
      const evalOut = model.evaluate(xTeT, yTeT) as ReturnType<typeof tf.tensor>[];
      const evalLoss = (await evalOut[0].data())[0];
      const accuracy = (await evalOut[1].data())[0];

      // 9. Extract weights in TF.js format [in, out] — server will transpose back to PyTorch
      const wts = model.getWeights();
      const layers: { kernel: number[][]; bias: number[] }[] = [];
      for (let i = 0; i < wts.length; i += 2) {
        layers.push({
          kernel: (await wts[i].array()) as number[][],
          bias:   (await wts[i + 1].array()) as number[],
        });
      }

      // 10. Submit ONLY the weight deltas — raw data never sent
      const res = await submitWeights(displayName, {
        layers,
        num_samples: numTrain,
        metrics: { train_loss: lastTrainLoss, eval_loss: evalLoss, accuracy },
      });
      setResult(res);
      setStatus(await fetchStatus());

      // Cleanup GPU memory.
      // Do NOT dispose wts — model.dispose() owns those tensors.
      // Disposing wts here would cause "already disposed" on the next run.
      [xTrT, yTrT, xTeT, yTeT, ...evalOut].forEach(t => t.dispose?.());
      model.dispose();

    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Training failed");
    } finally {
      setTraining(false);
      setProgress(null);
    }
  };

  // Step indicators
  const step1 = modelPublished ? "done" : "waiting";
  const step2 = result ? "done" : hasData && modelPublished ? "active" : "waiting";
  const step3 = result ? "done" : training ? "active" : "waiting";

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 px-6 py-4 flex items-center gap-3">
        <Link href="/" className="text-slate-400 hover:text-slate-700 text-sm">← Home</Link>
        <span className="text-slate-300">/</span>
        <span className="font-bold text-slate-900">🏥 {capitalName} Hospital</span>
        <span className="ml-auto text-xs bg-emerald-50 text-emerald-600 font-semibold px-3 py-1 rounded-full">Client</span>
      </header>

      <div className="max-w-2xl mx-auto px-6 py-10 space-y-6">

        {/* FL Steps */}
        <section className="bg-white rounded-2xl border border-slate-200 p-6">
          <h2 className="text-xs font-bold uppercase tracking-widest text-slate-400 mb-5">
            Federated Learning Flow
          </h2>
          <div className="space-y-1">
            <Step n={1} label="Fetch global model from server" status={step1} />
            <Step n={2} label="Load local patient data (stays in this browser tab)" status={step2} />
            <Step n={3} label="Train with TF.js & send only weight updates" status={step3} />
          </div>
        </section>

        {/* Data selection */}
        <section className="bg-white rounded-2xl border border-slate-200 p-6 space-y-4">
          <h2 className="text-xs font-bold uppercase tracking-widest text-slate-400">Patient Data</h2>

          <div className="flex gap-2">
            {(["sample", "upload"] as const).map(mode => (
              <button
                key={mode}
                onClick={() => setDataMode(mode)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  dataMode === mode
                    ? "bg-indigo-100 text-indigo-700"
                    : "bg-slate-100 text-slate-500 hover:bg-slate-200"
                }`}
              >
                {mode === "sample" ? "Sample dataset" : "Upload CSV"}
              </button>
            ))}
          </div>

          {dataMode === "sample" ? (
            <div className="space-y-2">
              <select
                value={dataset}
                onChange={e => setDataset(e.target.value)}
                className="w-full border border-slate-200 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400"
              >
                {SAMPLE_DATASETS.map(d => <option key={d.id} value={d.id}>{d.label}</option>)}
              </select>
              <p className="text-xs text-slate-400">
                Public UCI Heart Disease dataset — for demo purposes. In production each hospital loads its own private CSV locally.
              </p>
            </div>
          ) : (
            <>
              <input ref={fileRef} type="file" accept=".csv,.data" className="hidden"
                onChange={e => setFile(e.target.files?.[0] ?? null)} />
              <div
                onClick={() => fileRef.current?.click()}
                className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-colors ${
                  file ? "border-emerald-300 bg-emerald-50" : "border-slate-200 hover:border-indigo-300 hover:bg-indigo-50"
                }`}
              >
                {file ? (
                  <>
                    <p className="text-sm font-semibold text-emerald-700">{file.name}</p>
                    <p className="text-xs text-emerald-500 mt-1">{(file.size / 1024).toFixed(1)} KB · click to change</p>
                  </>
                ) : (
                  <>
                    <p className="text-sm text-slate-500">Click to browse CSV</p>
                    <p className="text-xs text-slate-400 mt-1">
                      14 columns (no header): age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, num
                    </p>
                  </>
                )}
              </div>
              <p className="text-xs text-emerald-600 font-medium">
                ✓ The CSV is read by the browser&apos;s FileReader API — it never leaves your machine.
              </p>
            </>
          )}
        </section>

        {/* Training config + action */}
        <section className="bg-white rounded-2xl border border-slate-200 p-6 space-y-4">
          <h2 className="text-xs font-bold uppercase tracking-widest text-slate-400">Local Training</h2>

          <div className="grid grid-cols-2 gap-4">
            <label className="space-y-1">
              <span className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Epochs</span>
              <input type="number" min={1} max={50} value={epochs}
                onChange={e => setEpochs(+e.target.value)}
                className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400" />
            </label>
            <label className="space-y-1">
              <span className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Learning Rate</span>
              <select value={lr} onChange={e => setLr(+e.target.value)}
                className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400">
                {[0.001, 0.01, 0.05, 0.1].map(v => <option key={v} value={v}>{v}</option>)}
              </select>
            </label>
          </div>

          {/* Progress bar */}
          {training && progress && (
            <div className="space-y-1">
              <div className="flex justify-between text-xs text-slate-500">
                <span>Epoch {progress.epoch} / {progress.total}</span>
                <span>loss {progress.loss.toFixed(4)}</span>
              </div>
              <div className="w-full bg-slate-100 rounded-full h-1.5">
                <div
                  className="bg-indigo-500 h-1.5 rounded-full transition-all"
                  style={{ width: `${(progress.epoch / progress.total) * 100}%` }}
                />
              </div>
            </div>
          )}

          <button
            disabled={!modelPublished || !hasData || training}
            onClick={handleTrain}
            className="w-full bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white font-semibold py-3 rounded-xl transition-colors"
          >
            {training ? "Training in browser…" : "Train & Submit Updates"}
          </button>

          {!modelPublished && (
            <p className="text-xs text-amber-600 text-center">
              Waiting for <Link href="/server" className="underline font-semibold">Central Server</Link> to publish the global model first.
            </p>
          )}
          {error && <p className="text-xs text-red-500 bg-red-50 rounded-lg px-3 py-2">{error}</p>}
        </section>

        {/* Training result */}
        {result && (
          <section className="bg-white rounded-2xl border border-slate-200 p-6">
            <h2 className="text-xs font-bold uppercase tracking-widest text-slate-400 mb-4">Training Results</h2>
            <div className="grid grid-cols-3 gap-4 mb-4">
              <Metric label="Accuracy" value={`${(result.metrics.accuracy * 100).toFixed(1)}%`} color="emerald" />
              <Metric label="Train Loss" value={result.metrics.train_loss.toFixed(4)} color="indigo" />
              <Metric label="Samples" value={String(result.num_samples)} color="slate" />
            </div>
            <p className="text-xs text-slate-400">
              Weight updates submitted. Raw patient data never left this browser tab.
              The Central Server can now include your updates in FedAvg aggregation.
            </p>
          </section>
        )}

        {/* Previous submission from server state */}
        {!result && client && client.rounds_submitted > 0 && (
          <section className="bg-white rounded-2xl border border-slate-200 p-6">
            <h2 className="text-xs font-bold uppercase tracking-widest text-slate-400 mb-4">Previous Submission</h2>
            <div className="grid grid-cols-3 gap-4 mb-4">
              <Metric label="Accuracy"
                value={client.metrics?.accuracy != null ? `${(client.metrics.accuracy * 100).toFixed(1)}%` : "—"}
                color="emerald" />
              <Metric label="Train Loss"
                value={client.metrics?.train_loss != null ? client.metrics.train_loss.toFixed(4) : "—"}
                color="indigo" />
              <Metric label="Samples"
                value={client.num_samples > 0 ? String(client.num_samples) : "—"}
                color="slate" />
            </div>
            <p className="text-xs text-slate-400">Rounds submitted: <strong>{client.rounds_submitted}</strong></p>
          </section>
        )}

        {isComplete && (
          <div className="bg-emerald-50 border border-emerald-200 rounded-2xl px-6 py-4 text-emerald-800 text-sm font-medium">
            ✓ Federation complete — the global model has been updated with contributions from all hospitals.
          </div>
        )}
      </div>
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

function Step({ n, label, status }: { n: number; label: string; status: "done" | "active" | "waiting" }) {
  const styles = {
    done:    { circle: "bg-emerald-100 text-emerald-700", icon: "✓", text: "text-slate-700" },
    active:  { circle: "bg-indigo-500 text-white",        icon: "●", text: "text-slate-900 font-semibold" },
    waiting: { circle: "bg-slate-100 text-slate-400",     icon: String(n), text: "text-slate-400" },
  }[status];

  return (
    <div className="flex items-center gap-3 py-2.5 border-b border-slate-50 last:border-0">
      <div className={`w-8 h-8 rounded-lg flex items-center justify-center text-sm font-bold shrink-0 ${styles.circle}`}>
        {styles.icon}
      </div>
      <span className={`text-sm ${styles.text}`}>{label}</span>
    </div>
  );
}

function Metric({ label, value, color }: { label: string; value: string; color: string }) {
  const bg = { emerald: "bg-emerald-50 text-emerald-700", indigo: "bg-indigo-50 text-indigo-700", slate: "bg-slate-50 text-slate-700" }[color] ?? "bg-slate-50 text-slate-700";
  return (
    <div className={`rounded-xl p-4 text-center ${bg}`}>
      <div className="text-xl font-black">{value}</div>
      <div className="text-xs font-semibold uppercase tracking-wide opacity-70 mt-1">{label}</div>
    </div>
  );
}
