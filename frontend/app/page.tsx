"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import Link from "next/link";
import { ThemeToggle } from "@/components/theme-toggle";
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
    <main className="min-h-screen bg-slate-50 dark:bg-[#0B101E] text-slate-600 dark:text-slate-400 selection:bg-yellow-500/30 overflow-x-hidden flex flex-col transition-colors duration-300">
      {/* Background ambient glow removed for matte look */}

      {/* Premium Edge-to-Edge Navigation */}
      <header className="sticky top-0 z-50 w-full bg-white/80 dark:bg-slate-900/60 backdrop-blur-2xl border-b border-cyan-50/80 dark:border-white/5 shadow-[0_8px_32px_rgba(30,27,75,0.04)] dark:shadow-[0_8px_32px_rgba(0,0,0,0.4)] px-6 sm:px-10 py-4 flex items-center justify-between transition-all duration-300">
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-yellow-600 to-stone-600 flex items-center justify-center shadow-lg shadow-yellow-600/20 dark:shadow-yellow-600/40">
            <Activity className="w-5 h-5 text-white" />
          </div>
          <span className="font-heading font-bold tracking-tight text-slate-900 dark:text-white text-lg">
            Federated<span className="text-slate-400 dark:text-slate-500 font-medium ml-1">Learning</span>
          </span>
        </div>
        
        <div className="flex items-center gap-4">
          <ThemeToggle />
          <Link 
            href="/docs" 
            className="group relative flex items-center gap-2.5 text-xs font-heading font-bold uppercase tracking-widest text-slate-700 dark:text-slate-200 transition-all px-5 py-3 rounded-full overflow-hidden bg-white/40 dark:bg-slate-800/40 hover:bg-slate-50 dark:hover:bg-slate-700 border border-transparent hover:border-slate-200 dark:hover:border-slate-600 backdrop-blur-md"
          >
            <span className="hidden sm:inline">Docs</span>
          </Link>
          <Link 
            href="/server" 
            className="group relative flex items-center gap-2.5 text-xs font-heading font-bold uppercase tracking-widest text-slate-700 dark:text-slate-200 transition-all px-6 py-3 rounded-full overflow-hidden bg-white dark:bg-slate-800 hover:bg-slate-50 dark:hover:bg-slate-700 border border-slate-200 dark:border-slate-700 hover:border-yellow-600/30 dark:hover:border-cyan-400/30 hover:shadow-md hover:shadow-yellow-600/10 dark:shadow-lg dark:shadow-black/20"
          >
            <Server className="w-4 h-4 text-yellow-700 dark:text-yellow-500 transition-transform duration-300 group-hover:scale-110" />
            <span className="hidden sm:inline">Server Dashboard</span>
          </Link>
        </div>
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
          <div className="inline-flex items-center gap-2 bg-yellow-50 dark:bg-slate-800 text-yellow-700 dark:text-yellow-600 text-[10px] font-heading font-bold uppercase tracking-widest px-4 py-1.5 rounded-full mb-6 relative border border-yellow-200 dark:border-slate-700">
            <span className="relative z-10 block pt-0.5">RESEARCH DEMO</span>
          </div>
          <h1 className="text-5xl lg:text-7xl font-heading font-bold text-slate-900 dark:text-white leading-[1.1] tracking-tight mb-6">
            Federated Learning <br className="hidden lg:block" />
            <span className="text-yellow-600">
              Hospital Simulation
            </span>
          </h1>
          <p className="text-slate-500 dark:text-slate-400 text-xl leading-relaxed max-w-2xl mx-auto">
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
            <div className="w-14 h-14 rounded-2xl bg-yellow-600 text-white flex items-center justify-center text-xl font-heading font-bold mb-5 border border-yellow-700/50">
              1
            </div>
            <h3 className="font-heading font-bold text-slate-900 dark:text-white mb-2.5">Fetch Global Model</h3>
            <p className="text-sm text-slate-500 dark:text-slate-400 leading-relaxed max-w-[200px]">Hospitals download the latest shared intelligence model from the central server.</p>
          </div>

          <div className="hidden md:flex pt-6 px-2 text-cyan-200 dark:text-slate-700">
            <ArrowRight className="w-5 h-5" />
          </div>

          {/* Step 2 */}
          <div className="flex-1 flex flex-col items-center text-center px-2">
            <div className="w-14 h-14 rounded-2xl bg-yellow-600 text-white flex items-center justify-center text-xl font-heading font-bold mb-5 border border-yellow-700/50">
              2
            </div>
            <h3 className="font-heading font-bold text-slate-900 dark:text-white mb-2.5">Load Private Data</h3>
            <p className="text-sm text-slate-500 dark:text-slate-400 leading-relaxed max-w-[200px]">Local patient records are loaded securely. Raw data never leaves the hospital.</p>
          </div>

          <div className="hidden md:flex pt-6 px-2 text-cyan-200 dark:text-slate-700">
            <ArrowRight className="w-5 h-5" />
          </div>

          {/* Step 3 */}
          <div className="flex-1 flex flex-col items-center text-center px-2">
            <div className="w-14 h-14 rounded-2xl bg-yellow-600 text-white flex items-center justify-center text-xl font-heading font-bold mb-5 border border-yellow-700/50">
              3
            </div>
            <h3 className="font-heading font-bold text-slate-900 dark:text-white mb-2.5">On-Device Training</h3>
            <p className="text-sm text-slate-500 dark:text-slate-400 leading-relaxed max-w-[200px]">The model learns from the private data locally, generating diagnostic weight updates.</p>
          </div>

          <div className="hidden md:flex pt-6 px-2 text-cyan-200 dark:text-slate-700">
            <ArrowRight className="w-5 h-5" />
          </div>

          {/* Step 4 */}
          <div className="flex-1 flex flex-col items-center text-center px-2">
            <div className="w-14 h-14 rounded-2xl bg-yellow-600 text-white flex items-center justify-center text-xl font-heading font-bold mb-5 border border-yellow-700/50">
              4
            </div>
            <h3 className="font-heading font-bold text-slate-900 dark:text-white mb-2.5">Secure Aggregation</h3>
            <p className="text-sm text-slate-500 dark:text-slate-400 leading-relaxed max-w-[200px]">Only the mathematical updates are sent back to the server to enhance the model.</p>
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
          <StatCard title="Connected Hospitals" value={activeNodes.toString()} icon={<Database />} />
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
          {/* Ambient Glow behind card removed for matte look */}
          
          <div className="relative bg-white/60 dark:bg-slate-900/60 backdrop-blur-2xl border border-white dark:border-slate-700/50 p-10 rounded-[2rem] shadow-[0_8px_32px_rgba(0,0,0,0.06)] dark:shadow-[0_8px_32px_rgba(0,0,0,0.3)] overflow-hidden">
            {/* Top Shine */}
            <div className="absolute top-0 inset-x-0 h-px bg-gradient-to-r from-transparent via-white dark:via-slate-500 to-transparent opacity-80" />
            
            <div className="flex flex-col items-center">
              <div className="flex items-center justify-center w-14 h-14 bg-gradient-to-br from-white to-emerald-50 dark:from-slate-800 dark:to-emerald-900/20 rounded-2xl mb-6 mx-auto border border-white dark:border-slate-700">
                <span className="text-2xl">🏥</span>
              </div>
              <h2 className="text-3xl font-heading font-bold text-slate-900 dark:text-white mb-3 tracking-tight">Hospital Portals</h2>
              <p className="text-slate-500 dark:text-slate-400 font-medium max-w-lg mx-auto md:mx-0 mb-10 text-center">
                Access a participating hospital's local training environment.
              </p>
            </div>

            <div className="space-y-10">
              <div className="grid grid-cols-2 gap-4">
                {PRESETS.map((p) => (
                  <button
                    key={p}
                    onClick={() => enterHospital(p)}
                    className="relative group px-5 py-4 rounded-2xl bg-white/40 dark:bg-slate-800/20 hover:bg-yellow-50 dark:hover:bg-yellow-600/10 border border-yellow-200/80 dark:border-yellow-600/40 hover:border-yellow-500 dark:hover:border-yellow-500/80 shadow-[0_2px_10px_rgba(202,138,4,0.04)] hover:shadow-[0_8px_30px_rgba(202,138,4,0.2)] transition-all duration-300 flex items-center justify-between backdrop-blur-md"
                  >
                    <span className="relative z-10 font-bold text-yellow-800/90 dark:text-yellow-500/90 group-hover:text-yellow-700 dark:group-hover:text-yellow-400 transition-colors">
                      {p}
                    </span>
                    <ArrowRight className="w-5 h-5 text-yellow-500/50 dark:text-yellow-600/50 group-hover:text-yellow-600 dark:group-hover:text-yellow-500 transition-transform group-hover:translate-x-1" />
                  </button>
                ))}
              </div>
              
              <div className="relative flex items-center py-2">
                <div className="flex-1 border-t border-slate-200/60 dark:border-slate-700/60" />
                <span className="px-4 text-xs font-heading font-bold uppercase tracking-widest text-slate-400 dark:text-slate-500 bg-transparent">Or Join Custom</span>
                <div className="flex-1 border-t border-slate-200/60 dark:border-slate-700/60" />
              </div>

              <div className="flex flex-col sm:flex-row gap-4 relative z-10">
                <input
                  type="text"
                  placeholder="Enter custom hospital name…"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && enterHospital(name)}
                  className="flex-1 bg-white/50 dark:bg-slate-800/30 hover:bg-yellow-50/50 dark:hover:bg-slate-800/60 focus:bg-white dark:focus:bg-slate-800/90 backdrop-blur-md border border-yellow-200/80 dark:border-yellow-600/40 rounded-2xl px-6 py-4 text-[15px] font-medium text-slate-900 dark:text-white focus:outline-none focus:border-yellow-500 focus:ring-4 focus:ring-yellow-500/20 shadow-inner transition-all placeholder:text-slate-400/80 dark:placeholder:text-slate-500"
                />
                <button
                  onClick={() => enterHospital(name)}
                  disabled={!name.trim()}
                  className="group relative sm:w-[160px] flex items-center justify-center overflow-hidden bg-yellow-600 dark:bg-yellow-600 disabled:bg-yellow-100 disabled:dark:bg-yellow-900/30 disabled:border disabled:border-yellow-200/60 dark:disabled:border-yellow-700/30 text-white font-bold tracking-wide px-6 py-4 rounded-2xl transition-all shadow-[0_8px_20px_rgba(202,138,4,0.15)] disabled:shadow-none disabled:text-yellow-700/40 dark:disabled:text-yellow-600/40 hover:shadow-[0_8px_25px_rgba(202,138,4,0.3)] hover:-translate-y-0.5 hover:bg-yellow-500 hover:dark:bg-yellow-500"
                >
                  <span className="relative z-10 flex items-center gap-2">
                    Join
                    <ArrowRight className="w-5 h-5 transition-transform group-hover:translate-x-1" />
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
    <div className="bg-white/50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-700 rounded-2xl p-6 flex flex-col items-center justify-center text-center shadow-sm">
      <div className="w-8 h-8 rounded-lg bg-yellow-50 dark:bg-slate-800 flex items-center justify-center text-yellow-700 dark:text-yellow-600 border border-yellow-100 dark:border-slate-700 mb-3">
        {icon}
      </div>
      <div className="text-4xl font-heading font-bold text-slate-900 dark:text-white mb-2 tracking-tight">{value}</div>
      <div className="text-xs font-heading font-bold uppercase tracking-widest text-slate-400 dark:text-slate-500">{title}</div>
    </div>
  );
}
