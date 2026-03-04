"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import Link from "next/link";
import { Server, Activity, ArrowRight, ShieldCheck, Database, FileDigit, Cpu } from "lucide-react";
import { fetchStatus, fetchResults, type FederationStatus } from "@/lib/api";
import { cn } from "@/lib/utils";

const PRESETS = ["Cleveland", "Hungarian", "Budapest", "Oslo"];

export default function Landing() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [status, setStatus] = useState<FederationStatus | null>(null);
  const [peakAccuracy, setPeakAccuracy] = useState<number | null>(null);

  useEffect(() => {
    async function loadStats() {
      try {
        const [s, r] = await Promise.all([fetchStatus(), fetchResults()]);
        setStatus(s);
        if (r && r.accuracy && r.accuracy.length > 0) {
          const validAccuracies = r.accuracy.filter((a): a is number => a !== null);
          if (validAccuracies.length > 0) {
            setPeakAccuracy(Math.max(...validAccuracies));
          }
        }
      } catch (e) {
        console.error("Failed to load global stats:", e);
      }
    }
    loadStats();
    // Poll slowly on the landing page just to keep stats relatively fresh
    const id = setInterval(loadStats, 10000);
    return () => clearInterval(id);
  }, []);

  const enterHospital = (n: string) => {
    if (!n.trim()) return;
    router.push(`/hospital/${encodeURIComponent(n.trim().toLowerCase())}`);
  };

  const activeNodes = Object.keys(status?.clients ?? {}).length;
  const currentIteration = status?.current_round ?? 0;

  return (
    <main className="min-h-screen bg-slate-50 text-slate-600 selection:bg-indigo-500/30 overflow-x-hidden flex flex-col">
      {/* Background ambient glow */}
      <div className="fixed top-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] bg-indigo-500/5 rounded-full blur-[100px] pointer-events-none" />

      {/* Premium Edge-to-Edge Navigation */}
      <header className="sticky top-0 z-50 w-full bg-white/80 backdrop-blur-2xl border-b border-indigo-50/80 shadow-[0_8px_32px_rgba(30,27,75,0.04)] px-6 sm:px-10 py-4 flex items-center justify-between transition-all duration-300">
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Activity className="w-5 h-5 text-white" />
          </div>
          <span className="font-heading font-bold tracking-tight text-slate-900 text-lg">
            Federated<span className="text-slate-400 font-medium ml-1">Learning</span>
          </span>
        </div>
        
        <Link 
          href="/server" 
          className="group relative flex items-center gap-2.5 text-xs font-heading font-bold uppercase tracking-widest text-slate-700 transition-all px-6 py-3 rounded-full overflow-hidden bg-white hover:bg-slate-50 border border-slate-200 hover:border-indigo-500/30 hover:shadow-md hover:shadow-indigo-500/10"
        >
          <Server className="w-4 h-4 text-indigo-600 transition-transform duration-300 group-hover:scale-110" />
          <span>Server Dashboard</span>
        </Link>
      </header>

      {/* Main Content */}
      <div className="flex-1 max-w-5xl mx-auto w-full px-6 py-16 lg:py-24 relative z-10 flex flex-col items-center">
        
        {/* Hero Section */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-16 max-w-4xl"
        >
          <div className="inline-flex items-center gap-2 bg-indigo-50 text-indigo-700 text-[10px] font-heading font-bold uppercase tracking-widest px-4 py-1.5 rounded-full mb-6 relative">
            <span className="relative z-10 block pt-0.5">RESEARCH DEMO</span>
          </div>
          <h1 className="text-5xl lg:text-7xl font-heading font-bold text-slate-900 leading-[1.1] tracking-tight mb-6">
            Federated Learning <br className="hidden lg:block" />
            <span className="text-indigo-600">
              Hospital Simulation
            </span>
          </h1>
          <p className="text-slate-500 text-xl leading-relaxed max-w-2xl mx-auto">
            Train a shared heart-disease prediction model across two hospitals —<br className="hidden md:block"/> without patient data ever leaving its source.
          </p>
        </motion.div>

        {/* 4 Steps Section */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="w-full flex flex-col md:flex-row items-start justify-center gap-4 md:gap-0 max-w-5xl mb-20 relative"
        >
          {/* Step 1 */}
          <div className="flex-1 flex flex-col items-center text-center px-2">
            <div className="w-14 h-14 rounded-2xl bg-indigo-500 text-white flex items-center justify-center text-xl font-heading font-bold mb-5 shadow-lg shadow-indigo-500/20">
              1
            </div>
            <h3 className="font-heading font-bold text-slate-900 mb-2.5">Fetch Global Model</h3>
            <p className="text-sm text-slate-500 leading-relaxed max-w-[200px]">Hospitals download the latest shared intelligence model from the central server.</p>
          </div>

          <div className="hidden md:flex pt-6 px-2 text-indigo-200">
            <ArrowRight className="w-5 h-5" />
          </div>

          {/* Step 2 */}
          <div className="flex-1 flex flex-col items-center text-center px-2">
            <div className="w-14 h-14 rounded-2xl bg-indigo-500 text-white flex items-center justify-center text-xl font-heading font-bold mb-5 shadow-lg shadow-indigo-500/20">
              2
            </div>
            <h3 className="font-heading font-bold text-slate-900 mb-2.5">Load Private Data</h3>
            <p className="text-sm text-slate-500 leading-relaxed max-w-[200px]">Local patient records are loaded securely. Raw data never leaves the hospital.</p>
          </div>

          <div className="hidden md:flex pt-6 px-2 text-indigo-200">
            <ArrowRight className="w-5 h-5" />
          </div>

          {/* Step 3 */}
          <div className="flex-1 flex flex-col items-center text-center px-2">
            <div className="w-14 h-14 rounded-2xl bg-indigo-500 text-white flex items-center justify-center text-xl font-heading font-bold mb-5 shadow-lg shadow-indigo-500/20">
              3
            </div>
            <h3 className="font-heading font-bold text-slate-900 mb-2.5">On-Device Training</h3>
            <p className="text-sm text-slate-500 leading-relaxed max-w-[200px]">The model learns from the private data locally, generating diagnostic weight updates.</p>
          </div>

          <div className="hidden md:flex pt-6 px-2 text-indigo-200">
            <ArrowRight className="w-5 h-5" />
          </div>

          {/* Step 4 */}
          <div className="flex-1 flex flex-col items-center text-center px-2">
            <div className="w-14 h-14 rounded-2xl bg-indigo-500 text-white flex items-center justify-center text-xl font-heading font-bold mb-5 shadow-lg shadow-indigo-500/20">
              4
            </div>
            <h3 className="font-heading font-bold text-slate-900 mb-2.5">Secure Aggregation</h3>
            <p className="text-sm text-slate-500 leading-relaxed max-w-[200px]">Only the mathematical updates are sent back to the server to enhance the model.</p>
          </div>
        </motion.div>

        {/* Live Global Stats Layout */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="w-full grid grid-cols-1 md:grid-cols-3 gap-6 mb-16"
        >
          <StatCard title="Global Iteration" value={currentIteration.toString()} icon={<Cpu />} />
          <StatCard title="Active Nodes" value={activeNodes.toString()} icon={<Database />} />
          <StatCard 
            title="Peak Accuracy" 
            value={peakAccuracy ? `${(peakAccuracy * 100).toFixed(1)}%` : "—"} 
            icon={<FileDigit />} 
          />
        </motion.div>

        {/* Hospital Entry Console */}
        <motion.div 
          initial={{ opacity: 0, scale: 0.95, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="relative w-full max-w-xl mx-auto"
        >
          {/* Ambient Glow behind card */}
          <div className="absolute -inset-1 bg-gradient-to-r from-indigo-500/10 via-emerald-500/10 to-violet-500/10 rounded-[2.5rem] blur-2xl" />
          
          <div className="relative bg-white/60 backdrop-blur-2xl border border-white p-10 rounded-[2rem] shadow-[0_8px_32px_rgba(0,0,0,0.06)] overflow-hidden">
            {/* Top Shine */}
            <div className="absolute top-0 inset-x-0 h-px bg-gradient-to-r from-transparent via-white to-transparent opacity-80" />
            
            <div className="flex flex-col items-center">
              <div className="flex items-center justify-center w-14 h-14 bg-gradient-to-br from-white to-emerald-50 rounded-2xl mb-6 mx-auto border border-white shadow-[0_4px_20px_rgba(16,185,129,0.15)]">
                <span className="text-2xl drop-shadow-sm">🏥</span>
              </div>
              <h2 className="text-3xl font-heading font-bold text-slate-900 mb-2 tracking-tight">Node Client Portal</h2>
              <p className="text-slate-500 text-base text-center mb-10 w-5/6 leading-relaxed">
                Select an active hospital node or initialize a new federated client.
              </p>
            </div>

            <div className="space-y-8">
              <div className="grid grid-cols-2 gap-4">
                {PRESETS.map((p) => (
                  <button
                    key={p}
                    onClick={() => enterHospital(p)}
                    className="relative group px-5 py-4 rounded-2xl bg-white/40 hover:bg-white/90 border border-white/60 hover:border-white shadow-[0_2px_10px_rgba(0,0,0,0.02)] hover:shadow-[0_8px_20px_rgba(99,102,241,0.08)] transition-all duration-300 font-semibold text-slate-700 text-sm flex items-center justify-center backdrop-blur-md"
                  >
                    <span className="relative z-10">{p}</span>
                  </button>
                ))}
              </div>
              
              <div className="relative flex items-center py-2">
                <div className="flex-1 border-t border-slate-200/60" />
                <span className="px-4 text-xs font-heading font-bold uppercase tracking-widest text-slate-400 bg-transparent">Or Join Custom</span>
                <div className="flex-1 border-t border-slate-200/60" />
              </div>

              <div className="flex flex-col sm:flex-row gap-3 relative z-10">
                <input
                  type="text"
                  placeholder="Enter custom hospital name…"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && enterHospital(name)}
                  className="flex-1 bg-white/50 hover:bg-white/80 focus:bg-white backdrop-blur-md border border-white rounded-2xl px-5 py-4 text-[15px] focus:outline-none focus:ring-2 focus:ring-indigo-500/30 shadow-[0_2px_10px_rgba(0,0,0,0.02)] transition-all placeholder:text-slate-400"
                />
                <button
                  onClick={() => enterHospital(name)}
                  disabled={!name.trim()}
                  className="group relative sm:w-[160px] flex items-center justify-center overflow-hidden bg-slate-900 disabled:bg-white/50 disabled:border disabled:border-slate-200 text-white font-semibold px-6 py-4 rounded-2xl transition-all shadow-[0_8px_20px_rgba(0,0,0,0.12)] disabled:shadow-none disabled:text-slate-400 hover:shadow-[0_8px_25px_rgba(99,102,241,0.4)] hover:-translate-y-0.5"
                >
                  {name.trim() ? (
                    <div className="absolute inset-0 bg-gradient-to-r from-indigo-600 to-violet-600 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                  ) : null}
                  <span className="relative z-10 flex items-center gap-2">
                    Join Node
                    <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
                  </span>
                </button>
              </div>
            </div>
          </div>
        </motion.div>
      </div>

    </main>
  );
}

function StatCard({ title, value, icon }: { title: string, value: string, icon: React.ReactNode }) {
  return (
    <div className="bg-white/50 border border-slate-200 rounded-2xl p-6 flex flex-col items-center justify-center text-center shadow-sm">
      <div className="w-8 h-8 rounded-lg bg-indigo-50 flex items-center justify-center text-indigo-600 border border-indigo-100 mb-3">
        {icon}
      </div>
      <div className="text-4xl font-heading font-bold text-slate-900 mb-2 tracking-tight">{value}</div>
      <div className="text-xs font-heading font-bold uppercase tracking-widest text-slate-400">{title}</div>
    </div>
  );
}
