"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";

const PRESETS = ["Cleveland", "Hungarian", "Budapest", "Oslo"];

export default function Landing() {
  const router = useRouter();
  const [name, setName] = useState("");

  const enterHospital = (n: string) => {
    if (!n.trim()) return;
    router.push(`/hospital/${encodeURIComponent(n.trim().toLowerCase())}`);
  };

  return (
    <main className="min-h-screen bg-linear-to-br from-slate-50 to-indigo-50 flex flex-col items-center justify-center px-6 py-16">
      {/* Hero */}
      <div className="text-center mb-14 max-w-2xl">
        <span className="inline-block bg-indigo-100 text-indigo-700 text-xs font-bold uppercase tracking-widest px-4 py-1.5 rounded-full mb-5">
          Research Demo
        </span>
        <h1 className="text-5xl font-black text-slate-900 leading-tight tracking-tight mb-4">
          Federated Learning<br />
          <span className="text-indigo-600">Hospital Network</span>
        </h1>
        <p className="text-slate-500 text-lg leading-relaxed">
          Train a shared heart-disease model across independent hospitals —
          patient data never leaves its source.
        </p>
      </div>

      {/* Role cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full max-w-3xl">
        {/* Central Server */}
        <div className="bg-white rounded-2xl border border-slate-200 p-8 shadow-sm hover:shadow-md hover:-translate-y-1 transition-all">
          <div className="w-12 h-12 bg-indigo-100 rounded-xl flex items-center justify-center text-2xl mb-4">🖥️</div>
          <h2 className="text-xl font-bold text-slate-900 mb-2">Central Server</h2>
          <p className="text-slate-500 text-sm leading-relaxed mb-6">
            Publish the global model, trigger FedAvg aggregation, and view
            results. Never sees raw patient data.
          </p>
          <span className="text-xs font-bold uppercase tracking-widest text-indigo-600 bg-indigo-50 px-3 py-1 rounded-full">
            Server
          </span>
          <div className="mt-6">
            <button
              onClick={() => router.push("/server")}
              className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 rounded-xl transition-colors"
            >
              Enter as Central Server
            </button>
          </div>
        </div>

        {/* Hospital */}
        <div className="bg-white rounded-2xl border border-slate-200 p-8 shadow-sm hover:shadow-md hover:-translate-y-1 transition-all">
          <div className="w-12 h-12 bg-emerald-100 rounded-xl flex items-center justify-center text-2xl mb-4">🏥</div>
          <h2 className="text-xl font-bold text-slate-900 mb-2">Hospital</h2>
          <p className="text-slate-500 text-sm leading-relaxed mb-4">
            Monitor your hospital&apos;s FL status. Training runs via a local
            Python script — patient data never leaves the machine.
          </p>
          <span className="text-xs font-bold uppercase tracking-widest text-emerald-600 bg-emerald-50 px-3 py-1 rounded-full">
            Client
          </span>
          <div className="mt-6 space-y-3">
            <div className="flex gap-2 flex-wrap">
              {PRESETS.map((p) => (
                <button
                  key={p}
                  onClick={() => enterHospital(p)}
                  className="text-sm px-3 py-1.5 rounded-lg border border-slate-200 hover:border-indigo-400 hover:text-indigo-700 transition-colors font-medium"
                >
                  {p}
                </button>
              ))}
            </div>
            <div className="flex gap-2">
              <input
                type="text"
                placeholder="Custom name…"
                value={name}
                onChange={(e) => setName(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && enterHospital(name)}
                className="flex-1 border border-slate-200 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400"
              />
              <button
                onClick={() => enterHospital(name)}
                disabled={!name.trim()}
                className="bg-emerald-600 hover:bg-emerald-700 disabled:opacity-40 text-white font-semibold px-5 py-2.5 rounded-xl transition-colors text-sm"
              >
                Go
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* How it works */}
      <div className="mt-16 max-w-3xl w-full">
        <p className="text-center text-xs font-bold uppercase tracking-widest text-slate-400 mb-6">How it works</p>
        <div className="grid grid-cols-5 gap-2 text-center">
          {[
            ["🖥️", "Server publishes", "Initial model parameters"],
            ["⬇️", "Hospital fetches", "Downloads global model"],
            ["🏋️", "Trains locally", "PyTorch on private data"],
            ["⬆️", "Sends weights", "No raw data — only deltas"],
            ["🔀", "FedAvg", "Stronger global model"],
          ].map(([icon, title, desc], i) => (
            <div key={i}>
              <div className="text-2xl mb-2">{icon}</div>
              <div className="text-xs font-bold text-slate-700 mb-1">{title}</div>
              <div className="text-xs text-slate-400 leading-snug">{desc}</div>
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}
