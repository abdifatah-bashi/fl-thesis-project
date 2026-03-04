"use client";
import { useState, useEffect } from "react";
import Link from "next/link";
import {
  publishModel, startAggregation, resetAll,
  fetchStatus, fetchResults,
  type FederationStatus, type Results,
} from "@/lib/api";

const STATUS_COLOR: Record<string, string> = {
  idle:        "bg-slate-100 text-slate-500",
  ready:       "bg-blue-100 text-blue-700",
  aggregating: "bg-indigo-100 text-indigo-700",
  complete:    "bg-emerald-100 text-emerald-700",
};

export default function ServerDashboard() {
  const [status, setStatus]   = useState<FederationStatus | null>(null);
  const [results, setResults] = useState<Results | null>(null);
  const [loading, setLoading] = useState("");
  const [rounds, setRounds]   = useState(3);
  const [epochs, setEpochs]   = useState(1);
  const [lr, setLr]           = useState(0.01);

  const poll = async () => {
    const [s, r] = await Promise.all([fetchStatus(), fetchResults()]);
    setStatus(s);
    setResults(r);
  };

  useEffect(() => {
    poll();
    const id = setInterval(poll, 2000);
    return () => clearInterval(id);
  }, []);

  const isAggregating = status?.status === "aggregating";
  const isComplete    = status?.status === "complete";
  const clients       = Object.entries(status?.clients ?? {});

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Link href="/" className="text-slate-400 hover:text-slate-700 text-sm">← Home</Link>
          <span className="text-slate-300">/</span>
          <span className="font-bold text-slate-900">🖥️ Central Server</span>
        </div>
        {status && (
          <span className={`text-xs font-bold uppercase tracking-widest px-3 py-1 rounded-full ${STATUS_COLOR[status.status] ?? "bg-slate-100 text-slate-500"}`}>
            {status.status}
          </span>
        )}
      </header>

      <div className="max-w-4xl mx-auto px-6 py-10 space-y-8">

        {/* Step 1 — Publish */}
        <section className="bg-white rounded-2xl border border-slate-200 p-6">
          <h2 className="text-xs font-bold uppercase tracking-widest text-slate-400 mb-4">Step 1 — Publish Global Model</h2>
          {status?.model_published ? (
            <div className="flex items-center gap-3 text-emerald-700 bg-emerald-50 rounded-xl px-4 py-3 text-sm font-medium">
              <span>✓</span>
              <span>Global model published — hospitals can now fetch it.</span>
              {status.published_at && (
                <span className="ml-auto text-emerald-500 font-normal">{new Date(status.published_at).toLocaleTimeString()}</span>
              )}
            </div>
          ) : (
            <div className="space-y-3">
              <p className="text-sm text-slate-500">
                Create the initial HeartDiseaseNet and make it available for hospitals to fetch.
              </p>
              <button
                disabled={!!loading}
                onClick={async () => {
                  setLoading("publish");
                  await publishModel();
                  await poll();
                  setLoading("");
                }}
                className="bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white font-semibold px-6 py-2.5 rounded-xl transition-colors text-sm"
              >
                {loading === "publish" ? "Publishing…" : "Publish Global Model"}
              </button>
            </div>
          )}
        </section>

        {/* Step 2 — Aggregate */}
        <section className="bg-white rounded-2xl border border-slate-200 p-6">
          <h2 className="text-xs font-bold uppercase tracking-widest text-slate-400 mb-4">Step 2 — Aggregate</h2>

          {isComplete ? (
            <div className="flex items-center gap-3 text-emerald-700 bg-emerald-50 rounded-xl px-4 py-3 text-sm font-medium">
              <span>✓</span>
              <span>All {status?.total_rounds} rounds complete.</span>
            </div>
          ) : isAggregating ? (
            <div className="space-y-3">
              <div className="flex items-center justify-between text-sm text-slate-700 font-medium">
                <span>Round {status?.current_round} / {status?.total_rounds}</span>
                <span className="text-indigo-600 animate-pulse">Aggregating…</span>
              </div>
              <div className="w-full bg-slate-100 rounded-full h-2">
                <div
                  className="bg-indigo-500 h-2 rounded-full transition-all"
                  style={{ width: `${((status?.current_round ?? 0) / (status?.total_rounds ?? 1)) * 100}%` }}
                />
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              {!status?.model_published && (
                <p className="text-sm text-slate-400">Publish the global model first.</p>
              )}
              {status?.model_published && clients.length === 0 && (
                <p className="text-sm text-slate-400">
                  Waiting for hospitals to submit weight updates.
                  Run <code className="bg-slate-100 px-1.5 py-0.5 rounded text-xs">python client/hospital_client.py --name cleveland</code> in a terminal.
                </p>
              )}
              <div className="grid grid-cols-3 gap-4">
                <label className="space-y-1">
                  <span className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Rounds</span>
                  <input type="number" min={1} max={20} value={rounds} onChange={e => setRounds(+e.target.value)}
                    className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400" />
                </label>
                <label className="space-y-1">
                  <span className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Epochs / round</span>
                  <input type="number" min={1} max={20} value={epochs} onChange={e => setEpochs(+e.target.value)}
                    className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400" />
                </label>
                <label className="space-y-1">
                  <span className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Learning rate</span>
                  <select value={lr} onChange={e => setLr(+e.target.value)}
                    className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400">
                    {[0.001, 0.01, 0.05, 0.1].map(v => <option key={v} value={v}>{v}</option>)}
                  </select>
                </label>
              </div>
              <button
                disabled={!!loading || !status?.model_published || clients.length === 0}
                onClick={async () => {
                  setLoading("agg");
                  await startAggregation(rounds, epochs, lr);
                  await poll();
                  setLoading("");
                }}
                className="bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white font-semibold px-6 py-2.5 rounded-xl transition-colors text-sm"
              >
                {loading === "agg" ? "Starting…" : "Start Aggregation"}
              </button>
            </div>
          )}
        </section>

        {/* Connected Hospitals */}
        {clients.length > 0 && (
          <section className="bg-white rounded-2xl border border-slate-200 p-6">
            <h2 className="text-xs font-bold uppercase tracking-widest text-slate-400 mb-4">
              Connected Hospitals ({clients.length})
            </h2>
            <div className="space-y-3">
              {clients.map(([name, info]) => (
                <div key={name} className="flex items-center gap-4 bg-slate-50 rounded-xl px-4 py-3">
                  <span className="text-lg">🏥</span>
                  <div className="flex-1">
                    <span className="font-semibold text-slate-900 capitalize">{name}</span>
                    <span className="text-xs text-slate-400 ml-2">{info.num_samples} samples</span>
                  </div>
                  {info.metrics?.accuracy != null && (
                    <span className="text-sm font-semibold text-emerald-700">{(info.metrics.accuracy * 100).toFixed(1)}%</span>
                  )}
                  {info.metrics?.train_loss != null && (
                    <span className="text-xs text-slate-400">loss {info.metrics.train_loss.toFixed(4)}</span>
                  )}
                  <span className="text-xs text-slate-400">round {info.rounds_submitted}</span>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Results */}
        {results && results.rounds.length > 0 && (
          <section className="bg-white rounded-2xl border border-slate-200 p-6">
            <h2 className="text-xs font-bold uppercase tracking-widest text-slate-400 mb-4">Results</h2>
            <div className="grid grid-cols-3 gap-4 mb-6">
              {results.accuracy.length > 0 && results.accuracy[results.accuracy.length-1] != null && (
                <div className="bg-emerald-50 rounded-xl p-4 text-center">
                  <div className="text-2xl font-black text-emerald-700">
                    {((results.accuracy[results.accuracy.length-1]! * 100)).toFixed(1)}%
                  </div>
                  <div className="text-xs text-emerald-500 font-semibold uppercase tracking-wide mt-1">Final Accuracy</div>
                </div>
              )}
              {results.train_loss.length > 0 && results.train_loss[results.train_loss.length-1] != null && (
                <div className="bg-indigo-50 rounded-xl p-4 text-center">
                  <div className="text-2xl font-black text-indigo-700">
                    {results.train_loss[results.train_loss.length-1]!.toFixed(4)}
                  </div>
                  <div className="text-xs text-indigo-500 font-semibold uppercase tracking-wide mt-1">Final Loss</div>
                </div>
              )}
              <div className="bg-slate-50 rounded-xl p-4 text-center">
                <div className="text-2xl font-black text-slate-700">{results.rounds.length}</div>
                <div className="text-xs text-slate-400 font-semibold uppercase tracking-wide mt-1">Rounds</div>
              </div>
            </div>

            {/* Per-round table */}
            <table className="w-full text-sm">
              <thead>
                <tr className="text-xs uppercase tracking-wide text-slate-400 border-b border-slate-100">
                  <th className="text-left py-2">Round</th>
                  <th className="text-right py-2">Accuracy</th>
                  <th className="text-right py-2">Train Loss</th>
                  <th className="text-right py-2">Eval Loss</th>
                </tr>
              </thead>
              <tbody>
                {results.rounds.map((r, i) => (
                  <tr key={r} className="border-b border-slate-50">
                    <td className="py-2 font-medium text-slate-700">{r}</td>
                    <td className="py-2 text-right text-emerald-700 font-semibold">
                      {results.accuracy[i] != null ? `${(results.accuracy[i]! * 100).toFixed(1)}%` : "—"}
                    </td>
                    <td className="py-2 text-right text-slate-500">
                      {results.train_loss[i] != null ? results.train_loss[i]!.toFixed(4) : "—"}
                    </td>
                    <td className="py-2 text-right text-slate-500">
                      {results.eval_loss[i] != null ? results.eval_loss[i]!.toFixed(4) : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}

        {/* Reset */}
        <div className="flex justify-end">
          <button
            onClick={async () => { await resetAll(); await poll(); }}
            className="text-sm text-slate-400 hover:text-red-500 transition-colors"
          >
            Reset everything
          </button>
        </div>
      </div>
    </div>
  );
}
