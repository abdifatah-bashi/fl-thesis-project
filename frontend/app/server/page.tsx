"use client";
import { useState, useEffect } from "react";
import Link from "next/link";
import { ThemeToggle } from "@/components/theme-toggle";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Activity, Globe2, Network, Cpu, Server, 
  Zap, CheckCircle2, ShieldCheck, Clock, LineChart 
} from "lucide-react";
import {
  publishModel, resetAll,
  fetchStatus, fetchResults,
  type FederationStatus, type Results,
} from "@/lib/api";
import { cn } from "@/lib/utils";

export default function ServerDashboard() {
  const [status, setStatus]   = useState<FederationStatus | null>(null);
  const [results, setResults] = useState<Results | null>(null);
  const [loading, setLoading] = useState("");

  const poll = async () => {
    try {
      const [s, r] = await Promise.all([fetchStatus(), fetchResults()]);
      setStatus(s);
      setResults(r);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    poll();
    const id = setInterval(poll, 2000);
    return () => clearInterval(id);
  }, []);

  const isAggregating = status?.status === "aggregating";
  const clients       = Object.entries(status?.clients ?? {});
  const modelPublished = status?.model_published ?? false;

  const latestAccuracy = results?.accuracy?.length 
    ? results.accuracy[results.accuracy.length - 1] 
    : null;
    
  const latestLoss = results?.train_loss?.length 
    ? results.train_loss[results.train_loss.length - 1] 
    : null;

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-[#0B101E] text-slate-600 dark:text-slate-300 selection:bg-yellow-500/30 overflow-x-hidden transition-colors duration-300">
      {/* Background ambient glow */}
      <div className="fixed top-0 left-1/2 -translate-x-1/2 w-[1000px] h-[500px] bg-yellow-500/5 dark:bg-yellow-500/10 rounded-full blur-[120px] pointer-events-none" />
      
      {/* Header */}
      <header className="sticky top-0 z-50 bg-white/80 dark:bg-slate-900/60 backdrop-blur-xl border-b border-slate-200 dark:border-white/5 px-6 py-4 flex items-center justify-between shadow-[0_8px_32px_rgba(30,27,75,0.04)] dark:shadow-[0_8px_32px_rgba(0,0,0,0.4)]">
        <div className="flex items-center gap-4">
          <Link href="/" className="text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white transition-colors flex items-center justify-center w-8 h-8 rounded-full bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 hover:bg-slate-200 dark:hover:bg-slate-700">
            <span className="sr-only">Home</span>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><path d="m15 18-6-6 6-6"/></svg>
          </Link>
          <div className="h-4 w-px bg-slate-200 dark:bg-slate-700" />
          <div className="flex items-center gap-2">
            <Server className="w-5 h-5 text-yellow-700 dark:text-yellow-500" />
            <span className="font-heading font-bold tracking-wide text-slate-900 dark:text-white">Server Dashboard</span>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <ThemeToggle />
          {status && (
            <div className="flex items-center gap-3">
              {isAggregating && (
                <span className="hidden sm:flex items-center gap-2 text-[10px] font-heading font-bold uppercase tracking-widest text-yellow-800 dark:text-yellow-400 bg-yellow-50 dark:bg-yellow-500/20 px-3 py-1.5 rounded-full border border-cyan-200 dark:border-yellow-600/30">
                  <Activity className="w-3.5 h-3.5 animate-pulse" />
                  Aggregating Round
                </span>
              )}
              <span className={cn(
                "hidden sm:flex items-center gap-1.5 text-[10px] font-heading font-bold uppercase tracking-widest px-3 py-1.5 rounded-full border",
                status.status === "ready" || status.status === "idle" 
                  ? "bg-emerald-50 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-500/20" 
                  : "bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 border-slate-200 dark:border-slate-700"
              )}>
                <span className={cn("w-1.5 h-1.5 rounded-full", status.status === "ready" || status.status === "idle" ? "bg-emerald-500 animate-pulse" : "bg-slate-500")} />
                System {status.status}
              </span>
            </div>
          )}
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-12 relative z-10">
        
        <AnimatePresence mode="wait">
          {!modelPublished ? (
            /* ZERO STATE: Not Published */
            <motion.div 
              key="zero-state"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 1.05, filter: "blur(10px)" }}
              className="flex flex-col items-center justify-center min-h-[60vh] max-w-xl mx-auto text-center space-y-8"
            >
              <div className="relative">
                <div className="absolute inset-0 bg-cyan-200 blur-[60px] opacity-50 animate-pulse" />
                <div className="w-24 h-24 bg-gradient-to-br from-yellow-600 to-stone-600 rounded-3xl p-0.5 shadow-xl shadow-yellow-600/20 rotate-3 transition-transform hover:rotate-6">
                  <div className="w-full h-full bg-white rounded-[22px] flex items-center justify-center">
                    <Globe2 className="w-10 h-10 text-yellow-700" />
                  </div>
                </div>
              </div>
              
              <div className="space-y-3">
                <h1 className="text-4xl font-heading font-bold text-slate-900 dark:text-white tracking-tight">Initialize Global Intelligence</h1>
                <p className="text-slate-500 dark:text-slate-400 text-lg leading-relaxed">
                  Mint the genesis iteration of the HeartDiseaseNet globally.<br/> Once initialized, hospitals can seamlessly fetch, train, and submit their learned weights.
                </p>
              </div>

              <button
                disabled={!!loading}
                onClick={async () => {
                  setLoading("publish");
                  await publishModel();
                  await poll();
                  setLoading("");
                }}
                className="group relative inline-flex items-center justify-center px-8 py-4 font-bold text-white transition-all duration-300 rounded-2xl disabled:opacity-50 disabled:cursor-not-allowed hover:scale-105 active:scale-95"
              >
                <div className="absolute inset-0 w-full h-full bg-gradient-to-r from-yellow-700 via-stone-500 to-yellow-700 rounded-2xl blur opacity-70 group-hover:opacity-100 transition-opacity duration-300 animate-[gradient_3s_linear_infinite] bg-[length:200%_auto]" />
                <div className="absolute inset-0 w-full h-full bg-gradient-to-r from-yellow-700 to-stone-600 rounded-2xl" />
                <span className="relative flex items-center gap-2">
                  {loading === "publish" ? (
                    <><Activity className="w-5 h-5 animate-pulse" /> Initializing Network…</>
                  ) : (
                    <><Zap className="w-5 h-5" /> Publish Genesis Model</>
                  )}
                </span>
              </button>
            </motion.div>
          ) : (
            /* LIVE DASHBOARD STATE */
            <motion.div 
              key="live-state"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-8"
            >
              <div className="flex items-end justify-between mb-10">
                <div>
                  <h1 className="text-4xl font-heading font-bold text-slate-900 dark:text-white tracking-tight mb-2">Live Command Center</h1>
                  <p className="text-slate-500 dark:text-slate-400 text-base">Monitoring globally distributed hospital parameters.</p>
                </div>
                <div className="flex items-center gap-2 px-4 py-2 bg-white dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700 shadow-sm rounded-xl">
                  <ShieldCheck className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
                  <span className="text-xs font-medium text-emerald-700 dark:text-emerald-400 tracking-wide">End-to-End Privacy Preserved</span>
                </div>
              </div>

              {/* KPI Top Row */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <DashCard 
                  title="Global Iteration" 
                  value={status?.current_round?.toString() ?? "0"} 
                  icon={<Cpu />} 
                  color="indigo"
                  sub={`${results?.rounds?.length ?? 0} total rounds completed`}
                />
                <DashCard 
                  title="Connected Hospitals" 
                  value={String(clients.length)} 
                  icon={<Network />} 
                  color="emerald"
                  sub="Independent hospitals connected"
                />
                <DashCard 
                  title="Peak Accuracy" 
                  value={latestAccuracy ? `${(latestAccuracy * 100).toFixed(1)}%` : "—"} 
                  icon={<LineChart />} 
                  color="violet"
                  sub={latestLoss ? `Loss at ${latestLoss.toFixed(4)}` : "Awaiting first round"}
                />
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Left Col: Connected Hospitals Feed */}
                <div className="lg:col-span-1 space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-heading font-bold uppercase tracking-widest text-slate-500 dark:text-slate-400">Live Hospital Feed</h3>
                    <span className="text-[10px] font-bold bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-500 dark:text-slate-400 px-2 py-0.5 rounded-full">{clients.length} ONLINE</span>
                  </div>

                  {clients.length === 0 ? (
                    <div className="bg-white/50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700 border-dashed rounded-2xl p-8 text-center">
                      <div className="w-12 h-12 bg-white dark:bg-slate-700/50 rounded-full flex items-center justify-center shadow-sm border border-slate-100 dark:border-slate-600 mx-auto mb-3">
                        <Clock className="w-5 h-5 text-slate-400 dark:text-slate-500" />
                      </div>
                      <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Awaiting hospital connections</p>
                      <p className="text-[10px] text-slate-400 dark:text-slate-500 mt-2 uppercase tracking-wide">Hospitals will appear securely</p>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      <AnimatePresence>
                        {clients.map(([name, info]) => (
                          <motion.div 
                            key={name}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            className="bg-white/80 dark:bg-slate-800/60 backdrop-blur-md border border-slate-200 dark:border-slate-700 rounded-2xl p-4 flex items-start gap-4 hover:border-cyan-200 dark:hover:border-yellow-600/50 hover:shadow-md transition-all group"
                          >
                            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-50 to-blue-50 dark:from-cyan-900/40 dark:to-blue-900/40 border border-yellow-100 dark:border-cyan-800/50 flex items-center justify-center shrink-0">
                              <span className="text-lg">🏥</span>
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center justify-between mb-1">
                                <span className="font-bold text-slate-900 dark:text-white capitalize truncate">{name}</span>
                                <span className="text-[10px] font-mono font-bold text-emerald-700 bg-emerald-50 px-1.5 py-px rounded border border-emerald-100">Sync</span>
                              </div>
                              <div className="flex items-center gap-2 text-xs text-slate-500 font-medium">
                                <span>{info.num_samples} patient refs</span>
                                <span>•</span>
                                <span>Iter {info.rounds_submitted}</span>
                              </div>
                            </div>
                          </motion.div>
                        ))}
                      </AnimatePresence>
                    </div>
                  )}
                </div>

                {/* Right Col: Historical Rounds Table */}
                <div className="lg:col-span-2 space-y-4">
                  <h3 className="text-sm font-heading font-bold uppercase tracking-widest text-slate-500 dark:text-slate-400">Historical Protocol Metrics</h3>
                  
                  <div className="bg-white/60 dark:bg-slate-800/40 backdrop-blur-xl border border-slate-200 dark:border-slate-700 rounded-3xl overflow-hidden shadow-lg shadow-slate-200/50 dark:shadow-none">
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm text-left">
                        <thead className="text-[10px] uppercase tracking-widest text-slate-500 dark:text-slate-400 bg-slate-100/50 dark:bg-slate-800/50 border-b border-slate-200 dark:border-slate-700">
                          <tr>
                            <th className="px-6 py-4 font-bold">Round Iteration</th>
                            <th className="px-6 py-4 font-bold text-right">Global Accuracy</th>
                            <th className="px-6 py-4 font-bold text-right">Train Loss</th>
                            <th className="px-6 py-4 font-bold text-right">Eval Loss</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100 dark:divide-slate-700/50">
                          {!results || results.rounds.length === 0 ? (
                            <tr>
                              <td colSpan={4} className="px-6 py-12 text-center text-slate-400 italic text-xs">
                                No aggregated rounds recorded yet. The table will populate as clients train.
                              </td>
                            </tr>
                          ) : (
                            // Render rounds in reverse (latest first)
                            [...results.rounds].reverse().map((r, revIndex) => {
                              const i = results.rounds.length - 1 - revIndex;
                              const acc = results.accuracy[i];
                              const tLoss = results.train_loss[i];
                              const eLoss = results.eval_loss[i];
                              const isLatest = i === results.rounds.length - 1;
                              
                              return (
                                <motion.tr 
                                  key={r}
                                  initial={{ opacity: 0, y: -10 }}
                                  animate={{ opacity: 1, y: 0 }}
                                  className={cn("hover:bg-slate-50 dark:hover:bg-slate-800/60 transition-colors", isLatest && "bg-slate-50 dark:bg-slate-800/60")}
                                >
                                  <td className="px-6 py-4 whitespace-nowrap">
                                    <div className="flex items-center gap-2">
                                      <span className="font-mono font-bold text-slate-900 dark:text-white">#{r}</span>
                                      {isLatest && <span className="text-[9px] font-bold uppercase tracking-wider text-yellow-800 dark:text-yellow-400 bg-yellow-100 dark:bg-cyan-900/50 px-1.5 py-0.5 rounded border border-cyan-200 dark:border-cyan-800/50">Latest</span>}
                                    </div>
                                  </td>
                                  <td className="px-6 py-4 text-right">
                                    <span className={cn("font-mono font-bold", acc != null && acc > 0.8 ? "text-emerald-600 dark:text-emerald-400" : "text-yellow-700 dark:text-yellow-500")}>
                                      {acc != null ? `${(acc * 100).toFixed(1)}%` : "—"}
                                    </span>
                                  </td>
                                  <td className="px-6 py-4 text-right font-mono text-slate-500 dark:text-slate-400">
                                    {tLoss != null ? tLoss.toFixed(4) : "—"}
                                  </td>
                                  <td className="px-6 py-4 text-right font-mono text-slate-500 dark:text-slate-400">
                                    {eLoss != null ? eLoss.toFixed(4) : "—"}
                                  </td>
                                </motion.tr>
                              );
                            })
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </div>

              {/* Reset Footer */}
              <div className="flex justify-end pt-8">
                 <button
                   onClick={async () => { await resetAll(); await poll(); }}
                   className="flex items-center gap-2 text-xs font-heading font-bold uppercase tracking-widest text-slate-600 hover:text-red-400 transition-colors"
                 >
                   <span className="w-2 h-2 rounded-full bg-red-500/50" />
                   Purge System Data & Reset
                 </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        
      </main>
    </div>
  );
}

// ── Components ────────────────────────────────────────────────────────────────

function DashCard({ title, value, icon, sub, color }: { title: string, value: string, icon: React.ReactNode, sub: string, color: "indigo" | "emerald" | "violet" }) {
  const themes = {
    indigo: "from-cyan-50 dark:from-cyan-900/40 to-white dark:to-slate-800 border-yellow-100 dark:border-cyan-800/50 text-yellow-700 dark:text-yellow-500",
    emerald: "from-emerald-50 dark:from-emerald-900/40 to-white dark:to-slate-800 border-emerald-100 dark:border-emerald-800/50 text-emerald-600 dark:text-emerald-400",
    violet: "from-blue-50 dark:from-blue-900/40 to-white dark:to-slate-800 border-blue-100 dark:border-blue-800/50 text-stone-500 dark:text-blue-400"
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white/70 dark:bg-slate-800/40 backdrop-blur-xl border border-slate-200 dark:border-slate-700 rounded-3xl p-6 relative overflow-hidden group hover:border-cyan-200 dark:hover:border-yellow-600/50 transition-colors shadow-lg shadow-slate-200/40 dark:shadow-none"
    >
      <div className={cn("absolute -top-10 -right-10 w-32 h-32 rounded-full blur-3xl opacity-10 pointer-events-none transition-opacity group-hover:opacity-20", 
        color === 'indigo' ? 'bg-yellow-500' : color === 'emerald' ? 'bg-emerald-500' : 'bg-stone-500')} 
      />
      
      <div className="flex items-center justify-between mb-4 relative z-10">
        <h3 className="text-xs font-heading font-bold uppercase tracking-widest text-slate-400 dark:text-slate-500">{title}</h3>
        <div className={cn("w-10 h-10 rounded-xl flex items-center justify-center border bg-gradient-to-b shadow-sm", themes[color])}>
          <div className="w-5 h-5">{icon}</div>
        </div>
      </div>
      
      <div className="relative z-10">
        <div className="text-5xl font-heading font-bold text-slate-900 dark:text-white tracking-tight mb-2">{value}</div>
        <div className="text-sm font-medium text-slate-500 dark:text-slate-400">{sub}</div>
      </div>
    </motion.div>
  );
}
