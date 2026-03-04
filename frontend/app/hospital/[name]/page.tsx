"use client";
import { use, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { ThemeToggle } from "@/components/theme-toggle";
import { motion, AnimatePresence } from "framer-motion";
import {
  ArrowLeft,
  CheckCircle2,
  ChevronDown,
  CloudDownload,
  Cpu,
  Database,
  FileSpreadsheet,
  UploadCloud,
  Network,
  Activity,
  Sparkles,
  Globe2,
} from "lucide-react";
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import {
  fetchStatus, fetchGlobalModelJson, fetchSampleData, submitWeights,
  type FederationStatus, type TrainResult,
} from "@/lib/api";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

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
  const [initialWeights, setInitialWeights] = useState<{ round: number; weights: number[] } | null>(null);
  const [updatedWeights, setUpdatedWeights] = useState<{ round: number; weights: number[] } | null>(null);
  const [currentRoundRef, setCurrentRoundRef] = useState<number>(-1);
  const fileRef                   = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const poll = async () => setStatus(await fetchStatus());
    poll();
    const id = setInterval(poll, 2000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    // If model is published and we haven't fetched the weights for THIS round yet
    if (status?.model_published && status.current_round !== currentRoundRef) {
      fetchGlobalModelJson()
        .then(m => {
          // Grab first 8 parameters from the first layer's kernel for display
          const w = m.layers[0].kernel.flat().slice(0, 8);
          if (currentRoundRef === -1) {
            setInitialWeights({ round: status.current_round, weights: w });
          } else {
            setUpdatedWeights({ round: status.current_round, weights: w });
          }
          setCurrentRoundRef(status.current_round);
        })
        .catch(console.error);
    }
  }, [status?.model_published, status?.current_round, currentRoundRef]);

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
    <div className="min-h-screen bg-gradient-to-br from-cyan-50/80 via-white to-emerald-50/80 dark:from-[#0B101E] dark:via-slate-900 dark:to-emerald-950/20 text-slate-800 dark:text-slate-200">
      {/* Premium Edge-to-Edge Navigation */}
      <header className="sticky top-0 z-50 w-full bg-white/80 dark:bg-slate-900/60 backdrop-blur-2xl border-b border-cyan-50/80 dark:border-white/5 shadow-[0_8px_32px_rgba(30,27,75,0.04)] dark:shadow-[0_8px_32px_rgba(0,0,0,0.4)] px-6 sm:px-10 py-4 flex items-center justify-between transition-all duration-300">
        <div className="flex items-center gap-4">
          <Link href="/" className="group flex items-center gap-2 px-3 py-2 rounded-full hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-all">
            <ArrowLeft className="w-4 h-4 text-slate-400 dark:text-slate-500 group-hover:text-yellow-700 dark:group-hover:text-yellow-500 group-hover:-translate-x-0.5 transition-all" />
            <span className="text-xs font-heading font-bold tracking-widest uppercase text-slate-500 dark:text-slate-400 group-hover:text-slate-800 dark:group-hover:text-slate-200">Back to Network</span>
          </Link>
          
          <div className="h-6 w-px bg-slate-200 dark:bg-slate-700" />
          
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-yellow-600 to-stone-500 flex items-center justify-center text-white shadow-lg shadow-yellow-600/20 dark:shadow-yellow-600/40">
              <Activity className="w-5 h-5" />
            </div>
            <span className="font-heading font-bold text-slate-900 dark:text-white tracking-tight text-lg">{capitalName} Portal</span>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <ThemeToggle />
          <div className="hidden sm:flex items-center gap-2 px-4 py-2 rounded-full bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-inner">
            <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 dark:bg-emerald-400 animate-pulse" />
            <span className="text-[10px] font-heading font-bold tracking-widest uppercase text-slate-500 dark:text-slate-400">System Online</span>
          </div>
        </div>
      </header>

      <div className="max-w-3xl mx-auto px-6 py-12 space-y-8">

        {/* FL Steps */}
        <motion.section 
          initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
          className="bg-white/80 dark:bg-slate-900/40 backdrop-blur-xl rounded-3xl border border-white dark:border-slate-800 p-8 shadow-xl shadow-slate-200/40 dark:shadow-none"
        >
          <div className="flex items-center gap-2 mb-6">
            <Network className="w-5 h-5 text-yellow-600" />
            <h2 className="text-sm font-heading font-bold uppercase tracking-widest text-slate-500 dark:text-slate-400">
              Federated Learning Flow
            </h2>
          </div>
          <div className="space-y-4">
            <Step n={1} label="Receive global diagnostic model" status={step1} icon={<CloudDownload />} >
               <AnimatePresence>
                 {modelPublished && (
                   <motion.div 
                     initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }}
                     className="mt-4 bg-slate-50/50 dark:bg-slate-800/20 border border-slate-200/60 dark:border-slate-700/40 rounded-2xl p-6 overflow-hidden relative"
                   >
                     <div className="absolute top-0 right-0 w-32 h-32 bg-yellow-50/50 dark:bg-yellow-900/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/3 pointer-events-none" />
                     
                     <div className="flex items-center justify-between mb-5 relative z-10">
                       <div className="flex items-center gap-2">
                         <Sparkles className="w-4 h-4 text-yellow-600" />
                         <span className="text-[11px] font-heading font-bold uppercase tracking-widest text-slate-600 dark:text-slate-300">Global Knowledge Acquired</span>
                       </div>
                       <div className="flex items-center gap-1.5 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 px-2.5 py-1 rounded-full shadow-sm">
                         <Database className="w-3 h-3 text-slate-400" />
                         <span className="text-[10px] font-bold text-slate-500 tracking-wider">~1.2 MB Fetched</span>
                       </div>
                     </div>
                     
                     <div className="grid grid-cols-1 md:grid-cols-2 gap-6 relative z-10 mb-5">
                       <div>
                         <div className="flex items-center gap-2 mb-1.5">
                           <Activity className="w-4 h-4 text-slate-400" />
                           <div className="text-[10px] font-heading uppercase font-bold text-slate-500 tracking-widest">Diagnostic Focus</div>
                         </div>
                         <div className="text-sm font-bold text-slate-800 dark:text-white">Heart Disease Analysis</div>
                         <div className="text-xs font-medium text-slate-500 dark:text-slate-400 mt-0.5">13 fundamental clinical metrics</div>
                       </div>

                       <div className="md:border-l border-slate-200 dark:border-slate-700/50 md:pl-6 pt-4 md:pt-0 border-t md:border-t-0">
                         <div className="flex items-center gap-2 mb-1.5">
                           <Globe2 className="w-4 h-4 text-slate-400" />
                           <div className="text-[10px] font-heading uppercase font-bold text-slate-500 tracking-widest">Model Version</div>
                         </div>
                         <div className="flex items-baseline gap-2">
                            <div className="text-sm font-bold text-slate-800 dark:text-white">Iteration {initialWeights?.round ?? status?.current_round ?? 0}</div>
                            <div className="text-[9px] font-bold text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/20 px-1.5 py-0.5 rounded tracking-wider border border-blue-100 dark:border-blue-800/30 uppercase">Ready to Learn</div>
                         </div>
                       </div>
                     </div>

                     {/* Parameter Preview */}
                     <div className="pt-4 border-t border-slate-200 dark:border-slate-700/50 relative z-10">
                        <div className="flex items-center justify-between mb-3">
                          <span className="text-[10px] font-heading uppercase font-bold text-slate-500 tracking-widest mt-1">Live Parameter Snapshot</span>
                          <span className="text-[10px] font-mono font-medium text-slate-400 dark:text-slate-500 bg-slate-100 dark:bg-slate-800 px-2 py-0.5 rounded">layer_0.kernel</span>
                        </div>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-4">
                           {initialWeights ? initialWeights.weights.map((w: number, idx: number) => {
                             const labels = ["Age", "Sex", "CP Type", "Rest BP", "Chol", "Fast BS", "Rest ECG", "Max HR"];
                             return (
                               <div key={idx} className="flex justify-between items-center py-2.5 px-4 bg-white dark:bg-slate-800/50 rounded-xl border border-slate-100 dark:border-slate-700/50">
                                 <span className="text-[11px] font-bold text-slate-500 dark:text-slate-400 truncate max-w-[60px] md:max-w-none" title={labels[idx]}>{labels[idx]}</span>
                                 <span className={cn(
                                   "font-mono text-[11px] font-bold ml-2",
                                   w > 0 ? "text-emerald-600 dark:text-emerald-400" : "text-slate-600 dark:text-slate-400"
                                 )}>
                                   {w > 0 ? "+" : ""}{w.toFixed(4)}
                                 </span>
                               </div>
                             );
                           }) : (
                             <div className="col-span-4 text-center py-3 animate-pulse text-[11px] font-mono text-slate-400">Loading tensor data...</div>
                           )}
                        </div>
                     </div>
                   </motion.div>
                 )}
               </AnimatePresence>
            </Step>
            <Step n={2} label="Load local patient data (stays on device)" status={step2} icon={<Database />} />
            <Step n={3} label="Train locally & submit weight updates" status={step3} icon={<Cpu />} />
          </div>
        </motion.section>

        {/* Data selection */}
        <motion.section 
          initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}
          className="bg-white/80 dark:bg-slate-900/40 backdrop-blur-xl rounded-3xl border border-white dark:border-slate-800 p-8 shadow-xl shadow-slate-200/40 dark:shadow-none space-y-6"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <FileSpreadsheet className="w-5 h-5 text-emerald-500" />
              <h2 className="text-sm font-heading font-bold uppercase tracking-widest text-slate-500 dark:text-slate-400">Patient Data</h2>
            </div>
            <div className="flex bg-slate-100/80 dark:bg-slate-900/80 p-1 rounded-xl shadow-inner border border-slate-200/50 dark:border-slate-800/50">
              {(["sample", "upload"] as const).map(mode => (
                <button
                  key={mode}
                  onClick={() => setDataMode(mode)}
                  className={cn(
                    "relative px-4 py-2 rounded-lg text-sm font-semibold transition-all duration-300",
                    dataMode === mode ? "text-yellow-800 dark:text-yellow-500" : "text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300"
                  )}
                >
                  {dataMode === mode && (
                    <motion.div layoutId="dataModeTab" className="absolute inset-0 bg-white dark:bg-slate-800 rounded-lg shadow-sm border border-slate-100 dark:border-slate-700" />
                  )}
                  <span className="relative z-10">{mode === "sample" ? "Sample Data" : "Upload CSV"}</span>
                </button>
              ))}
            </div>
          </div>

          <AnimatePresence mode="wait">
            {dataMode === "sample" ? (
              <motion.div key="sample" initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.98 }} transition={{ duration: 0.2 }} className="space-y-3">
                <div className="relative">
                  <select
                    value={dataset}
                    onChange={e => setDataset(e.target.value)}
                    className="w-full appearance-none bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-2xl pl-4 pr-10 py-3.5 text-sm font-medium text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-4 focus:ring-yellow-600/10 focus:border-cyan-300 dark:focus:border-yellow-600/50 transition-all shadow-sm"
                  >
                    {SAMPLE_DATASETS.map(d => <option key={d.id} value={d.id}>{d.label}</option>)}
                  </select>
                  <ChevronDown className="absolute right-4 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400 pointer-events-none" />
                </div>
                <p className="text-xs text-slate-400 dark:text-slate-500 font-medium px-1">
                  Public UCI Heart Disease dataset — for demo purposes. In production, hospitals load private CSVs locally.
                </p>
              </motion.div>
            ) : (
              <motion.div key="upload" initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.98 }} transition={{ duration: 0.2 }}>
                <input ref={fileRef} type="file" accept=".csv,.data" className="hidden"
                  onChange={e => setFile(e.target.files?.[0] ?? null)} />
                <div
                  onClick={() => fileRef.current?.click()}
                  className={cn(
                    "border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-300 group",
                    file ? "border-emerald-400 bg-emerald-50/50 dark:bg-emerald-500/10 dark:border-emerald-500/50 shadow-inner" : "border-slate-300 dark:border-slate-700 hover:border-cyan-400 dark:hover:border-yellow-600/50 hover:bg-yellow-50/50 dark:hover:bg-yellow-500/5"
                  )}
                >
                  {file ? (
                    <div className="flex flex-col items-center justify-center gap-2">
                      <div className="w-12 h-12 bg-emerald-100 dark:bg-emerald-900/40 text-emerald-600 dark:text-emerald-400 rounded-full flex items-center justify-center mb-1 shadow-sm">
                        <CheckCircle2 className="w-6 h-6" />
                      </div>
                      <p className="text-sm font-bold text-emerald-800 dark:text-emerald-400">{file.name}</p>
                      <p className="text-xs text-emerald-600 dark:text-emerald-500 font-medium bg-emerald-100/50 dark:bg-emerald-900/30 px-3 py-1 rounded-full">{(file.size / 1024).toFixed(1)} KB · Click to change file</p>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center gap-2">
                      <div className="w-12 h-12 bg-slate-100 dark:bg-slate-800 text-slate-400 dark:text-slate-500 group-hover:bg-yellow-100 dark:group-hover:bg-cyan-900/40 group-hover:text-yellow-600 dark:group-hover:text-yellow-500 rounded-full flex items-center justify-center mb-1 transition-colors">
                        <UploadCloud className="w-6 h-6" />
                      </div>
                      <p className="text-sm font-bold text-slate-700 dark:text-slate-300 group-hover:text-yellow-800 dark:group-hover:text-yellow-500 transition-colors">Click to browse or drag CSV</p>
                      <p className="text-xs text-slate-400 dark:text-slate-500 mt-1 max-w-sm mx-auto">
                        14 columns (no header): age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, num
                      </p>
                    </div>
                  )}
                </div>
                <div className="mt-3 flex items-start gap-2 text-emerald-700 dark:text-emerald-400 bg-gradient-to-r from-emerald-50 to-emerald-50/30 dark:from-emerald-900/20 dark:to-emerald-900/10 p-3.5 rounded-xl border border-emerald-100/50 dark:border-emerald-800/30">
                  <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5 text-emerald-500 dark:text-emerald-400" />
                  <p className="text-xs font-semibold leading-relaxed">
                    Data privacy guaranteed. The CSV is read entirely by the browser&apos;s FileReader API and never leaves your machine.
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.section>

        {/* Training config + action */}
        <motion.section 
          initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}
          className="bg-white/80 dark:bg-slate-900/40 backdrop-blur-xl rounded-3xl border border-white dark:border-slate-800 p-8 shadow-xl shadow-slate-200/40 dark:shadow-none space-y-6"
        >
          <div className="flex items-center gap-2 mb-2">
            <Cpu className="w-5 h-5 text-stone-500" />
            <h2 className="text-sm font-heading font-bold uppercase tracking-widest text-slate-500 dark:text-slate-400">Local Training</h2>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <label className="space-y-1.5 focus-within:text-yellow-700 dark:focus-within:text-yellow-500 transition-colors group">
              <span className="text-[11px] font-heading font-bold text-slate-400 dark:text-slate-500 group-focus-within:text-yellow-600 dark:group-focus-within:text-yellow-500 uppercase tracking-widest block ml-1 transition-colors">Epochs</span>
              <input type="number" min={1} max={50} value={epochs}
                onChange={e => setEpochs(+e.target.value)}
                className="w-full bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl px-4 py-3 text-sm font-bold text-slate-700 dark:text-slate-200 focus:outline-none focus:ring-4 focus:ring-yellow-600/10 focus:border-cyan-400 dark:focus:border-yellow-600 transition-all shadow-sm" />
            </label>
            <label className="space-y-1.5 focus-within:text-yellow-700 dark:focus-within:text-yellow-500 transition-colors relative group">
              <span className="text-[11px] font-heading font-bold text-slate-400 dark:text-slate-500 group-focus-within:text-yellow-600 dark:group-focus-within:text-yellow-500 uppercase tracking-widest block ml-1 transition-colors">Learning Rate</span>
              <div className="relative">
                <select value={lr} onChange={e => setLr(+e.target.value)}
                  className="w-full appearance-none bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl pl-4 pr-10 py-3 text-sm font-bold text-slate-700 dark:text-slate-200 focus:outline-none focus:ring-4 focus:ring-yellow-600/10 focus:border-cyan-400 dark:focus:border-yellow-600 transition-all shadow-sm">
                  {[0.001, 0.01, 0.05, 0.1].map(v => <option key={v} value={v}>{v}</option>)}
                </select>
                <ChevronDown className="absolute right-4 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400 pointer-events-none" />
              </div>
            </label>
          </div>

          {/* Progress bar */}
          <AnimatePresence>
            {training && progress && (
              <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }} className="space-y-2 overflow-hidden py-2">
                <div className="flex justify-between text-[11px] font-bold text-slate-500 dark:text-slate-400 px-1 uppercase tracking-wide">
                  <span className="text-yellow-700 dark:text-yellow-500">Epoch {progress.epoch} of {progress.total}</span>
                  <span>Loss <span className="text-slate-700 dark:text-slate-300 font-mono ml-1">{progress.loss.toFixed(4)}</span></span>
                </div>
                <div className="w-full bg-slate-100 dark:bg-slate-800/50 rounded-full h-3 shadow-inner overflow-hidden relative border border-slate-200/50 dark:border-slate-700/50">
                  <motion.div
                    className="absolute top-0 left-0 bottom-0 bg-gradient-to-r from-yellow-600 via-stone-500 to-yellow-600 rounded-full w-full bg-[length:200%_auto] animate-[gradient_2s_linear_infinite]"
                    initial={{ scaleX: 0, originX: 0 }}
                    animate={{ scaleX: progress.epoch / progress.total }}
                    transition={{ type: "spring", bounce: 0, duration: 0.5 }}
                  />
                  <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI4IiBoZWlnaHQ9IjgiPgo8cmVjdCB3aWR0aD0iOCIgaGVpZ2h0PSI4IiBmaWxsPSIjZmZmIiBmaWxsLW9wYWNpdHk9IjAuMSI+PC9yZWN0Pgo8cGF0aCBkPSJNMCAwTDggOFoiIHN0cm9rZT0iI2ZmZiIHN0cm9rZS1vcGFjaXR5PSIwLjIiIHN0cm9rZS13aWR0aD0iMiI+PC9wYXRoPjwvc3ZnPg==')] opacity-30 animate-[slide_1s_linear_infinite]" />
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          <button
            disabled={!modelPublished || !hasData || training}
            onClick={handleTrain}
            className={cn(
              "group relative w-full rounded-2xl disabled:opacity-50 disabled:cursor-not-allowed transition-all hover:scale-[1.01] active:scale-[0.99] shadow-lg shadow-yellow-600/20",
              training ? "bg-stone-500" : "bg-yellow-600 hover:bg-yellow-500"
            )}
          >
            <div className="relative px-6 py-4 flex items-center justify-center gap-2 text-white font-bold tracking-wide">
              {training ? (
                <>
                  <Activity className="w-5 h-5 animate-pulse" />
                  Training Locally…
                </>
              ) : (
                <>
                  <CloudDownload className="w-5 h-5 rotate-180 group-hover:-translate-y-1 transition-transform" />
                  Train & Submit Updates
                </>
              )}
            </div>
          </button>

          {!modelPublished && (
            <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-xs text-yellow-800 dark:text-yellow-500 font-medium text-center bg-yellow-50 dark:bg-cyan-900/20 rounded-xl py-3 border border-cyan-200/50 dark:border-cyan-900/50 shadow-sm">
              Waiting for <Link href="/server" className="underline font-bold text-yellow-800 dark:text-yellow-600 hover:text-cyan-900 dark:hover:text-yellow-500">Central Server</Link> to publish the global model.
            </motion.p>
          )}
          {error && (
            <motion.p initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="text-xs text-red-600 dark:text-red-400 font-bold bg-red-50 dark:bg-red-900/20 rounded-xl px-4 py-3 border border-red-200/50 dark:border-red-900/50 shadow-sm flex items-start gap-2">
              <span className="text-red-500 text-lg leading-none mt-0.5">!</span> <span className="mt-1">{error}</span>
            </motion.p>
          )}
        </motion.section>

        {/* Training result */}
        <AnimatePresence>
          {result && (
            <motion.section 
              initial={{ opacity: 0, y: 20, scale: 0.95 }} animate={{ opacity: 1, y: 0, scale: 1 }}
              className="bg-white/90 dark:bg-slate-900/40 backdrop-blur-2xl rounded-3xl border border-emerald-100 dark:border-slate-800 p-8 shadow-2xl shadow-emerald-500/15 dark:shadow-none relative overflow-hidden"
            >
              {/* Premium Result Glow */}
              <div className="absolute top-0 right-0 w-64 h-64 bg-emerald-400/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/3 pointer-events-none" />
              
              <div className="relative z-10">
                <div className="flex items-center gap-3 mb-8">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-emerald-400 to-emerald-500 flex items-center justify-center shadow-lg shadow-emerald-500/30">
                    <CheckCircle2 className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h2 className="text-sm font-heading font-bold uppercase tracking-widest text-emerald-800 dark:text-emerald-400">Training Complete</h2>
                    <p className="text-sm text-slate-500 dark:text-slate-400 font-medium mt-0.5">Local round successfully finished and verified.</p>
                  </div>
                </div>
                
                <div className="grid grid-cols-3 gap-6 mb-8">
                  <Metric label="Accuracy Model" value={`${(result.metrics.accuracy * 100).toFixed(1)}%`} color="emerald" delay={0} />
                  <Metric label="Final Train Loss" value={result.metrics.train_loss.toFixed(4)} color="indigo" delay={0.1} />
                  <Metric label="Patient Records" value={String(result.num_samples)} color="violet" delay={0.2} />
                </div>
                
                <div className="bg-gradient-to-r from-emerald-50 to-white dark:from-emerald-900/20 dark:to-slate-800/80 rounded-2xl p-5 border border-emerald-100 dark:border-emerald-800/30 shadow-inner flex gap-4 items-center mb-6">
                  <div className="bg-white dark:bg-slate-800 p-2.5 rounded-xl shadow-sm border border-emerald-50 dark:border-emerald-800/50 shrink-0">
                    <Database className="w-6 h-6 text-emerald-600 dark:text-emerald-400" />
                  </div>
                  <p className="text-sm text-slate-700 dark:text-slate-300 font-medium leading-relaxed">
                    <strong className="text-emerald-900 dark:text-emerald-400 block mb-1">Privacy Preserved Submission</strong>
                    Weight deltas submitted successfully to the central server. Raw patient data never left your device. Local state is safely stored.
                  </p>
                </div>
                
                {updatedWeights && (
                  <div className="mt-4 border-t border-emerald-100/50 pt-6">
                    <ParameterSnapshot 
                      weights={updatedWeights.weights} 
                      label={`Updated Global Parameters (Iter ${updatedWeights.round})`} 
                    />
                  </div>
                )}
              </div>
            </motion.section>
          )}
        </AnimatePresence>

        {/* Previous submission from server state */}
        {!result && client && client.rounds_submitted > 0 && (
          <motion.section initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="bg-white/60 dark:bg-slate-800/40 backdrop-blur-xl rounded-3xl border border-white dark:border-slate-700/50 p-8 shadow-lg shadow-slate-200/30 dark:shadow-none">
            <h2 className="text-[11px] font-heading font-bold uppercase tracking-widest text-slate-400 dark:text-slate-500 mb-6">Historical Protocol Data</h2>
            <div className="grid grid-cols-3 gap-6 mb-5">
              <Metric label="Last Accuracy"
                value={client.metrics?.accuracy != null ? `${(client.metrics.accuracy * 100).toFixed(1)}%` : "—"}
                color="emerald" delay={0} />
              <Metric label="Last Train Loss"
                value={client.metrics?.train_loss != null ? client.metrics.train_loss.toFixed(4) : "—"}
                color="indigo" delay={0.05} />
              <Metric label="Total Samples"
                value={client.num_samples > 0 ? String(client.num_samples) : "—"}
                color="slate" delay={0.1} />
            </div>
            <div className="flex items-center justify-center mt-6">
              <div className="bg-slate-50 dark:bg-slate-800/80 border border-slate-200/60 dark:border-slate-700/60 rounded-full px-5 py-2 flex items-center gap-3 shadow-sm">
                <span className="relative flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-yellow-600 dark:bg-yellow-500 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-yellow-500 dark:bg-yellow-700"></span>
                </span>
                <span className="text-sm font-semibold text-slate-600 dark:text-slate-300">
                  Total Participated Rounds: <strong className="font-heading text-slate-900 dark:text-white ml-1 text-base">{client.rounds_submitted}</strong>
                </span>
              </div>
            </div>
          </motion.section>
        )}

        <AnimatePresence>
          {isComplete && (
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="bg-gradient-to-r from-emerald-500 to-emerald-400 rounded-2xl p-1 shadow-lg shadow-emerald-500/20">
              <div className="bg-white/95 dark:bg-slate-900/95 rounded-xl px-6 py-5 flex items-start gap-4">
                <div className="bg-emerald-100 dark:bg-emerald-900/40 rounded-full p-2 shrink-0">
                  <CheckCircle2 className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
                </div>
                <div>
                  <h4 className="font-heading font-bold text-slate-900 dark:text-white text-sm">Federation Complete</h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mt-1 font-medium">The global model has been updated with contributions from all hospitals.</p>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* To give some bottom padding */}
        <div className="h-4" />
      </div>
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

function Step({ n, label, status, icon, children }: { n: number; label: string; status: "done" | "active" | "waiting"; icon: React.ReactNode; children?: React.ReactNode }) {
  const styles = {
    done:    { circle: "bg-emerald-100 dark:bg-emerald-900/40 text-emerald-600 dark:text-emerald-400 shadow-inner", ring: "ring-emerald-100 dark:ring-emerald-900/20", text: "text-slate-700 dark:text-slate-200 font-semibold" },
    active:  { circle: "bg-gradient-to-tr from-yellow-600 to-stone-500 text-white shadow-md shadow-yellow-600/30", ring: "ring-cyan-100 dark:ring-cyan-900/40", text: "text-slate-900 dark:text-white font-bold" },
    waiting: { circle: "bg-slate-100 dark:bg-slate-800 text-slate-400 dark:text-slate-500", ring: "ring-transparent", text: "text-slate-400 dark:text-slate-500 font-medium" },
  }[status];

  return (
    <div className="flex items-start gap-5 py-3.5 group">
      <div className={cn(
        "relative w-11 h-11 rounded-2xl flex items-center justify-center shrink-0 transition-all duration-500 ring-4 mt-0.5",
        styles.circle, styles.ring,
        status === "active" && "scale-105"
      )}>
        {status === "done" ? <CheckCircle2 className="w-6 h-6" /> : (status === "active" ? <div className="w-5 h-5">{icon}</div> : <span className="text-sm font-bold">{n}</span>)}
        {status === "active" && (
           <div className="absolute -top-2 -right-2 w-5 h-5 bg-white dark:bg-slate-800 rounded-full flex items-center justify-center shadow-sm">
             <span className="text-[10px] font-bold text-yellow-700 dark:text-yellow-500">{n}</span>
           </div>
        )}
      </div>
      <div className="flex-1 min-w-0 pt-2.5">
        <span className={cn("text-sm transition-colors duration-300 block", styles.text)}>{label}</span>
        {children}
      </div>
    </div>
  );
}

function Metric({ label, value, color, delay = 0 }: { label: string; value: string; color: string; delay?: number }) {
  const themes: Record<string, string> = { 
    emerald: "from-emerald-50 dark:from-emerald-900/20 to-white dark:to-slate-800 text-emerald-700 dark:text-emerald-400 border-emerald-200/50 dark:border-emerald-800/50 shadow-emerald-500/10 dark:shadow-none", 
    indigo: "from-cyan-50 dark:from-cyan-900/20 to-white dark:to-slate-800 text-yellow-800 dark:text-yellow-500 border-cyan-200/50 dark:border-cyan-800/50 shadow-yellow-600/10 dark:shadow-none", 
    violet: "from-blue-50 dark:from-blue-900/20 to-white dark:to-slate-800 text-stone-600 dark:text-blue-400 border-blue-200/50 dark:border-blue-800/50 shadow-stone-500/10 dark:shadow-none",
    slate: "from-slate-50 dark:from-slate-800/50 to-white dark:to-slate-800 text-slate-700 dark:text-slate-300 border-slate-200/50 dark:border-slate-700/50 shadow-slate-500/5 dark:shadow-none" 
  };
  const theme = themes[color] ?? themes.slate;

  return (
    <motion.div 
      initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} transition={{ delay, duration: 0.5, type: "spring", bounce: 0.4 }}
      className={cn("rounded-2xl p-6 relative overflow-hidden bg-gradient-to-b border shadow-lg group hover:shadow-xl transition-shadow duration-300", theme)}
    >
      {/* Decorative background element */}
      <div className={cn(
        "absolute -right-4 -top-4 w-24 h-24 rounded-full opacity-10 blur-xl group-hover:opacity-20 transition-opacity duration-500",
        color === "emerald" ? "bg-emerald-500" : color === "indigo" ? "bg-yellow-500" : color === "violet" ? "bg-stone-500" : "bg-slate-400"
      )} />
      
      <div className="relative z-10 flex flex-col items-center">
        <div className="text-5xl font-heading font-bold tracking-tight mb-2">{value}</div>
        <div className="text-xs font-heading font-bold uppercase tracking-widest opacity-70 bg-white/50 dark:bg-slate-900/50 px-3 py-1 rounded-full backdrop-blur-sm shadow-sm">{label}</div>
      </div>
    </motion.div>
  );
}

const WEIGHT_LABELS = ["Age", "Sex", "CP Type", "Rest BP", "Chol", "Fast BS", "Rest ECG", "Max HR"];

function ParameterSnapshot({ weights, label }: { weights: number[] | null, label: string }) {
  return (
    <div className="mt-8 bg-slate-50/50 dark:bg-slate-800/30 border border-slate-200/60 dark:border-slate-700/50 rounded-2xl p-6 shadow-sm">
      <div className="flex items-center justify-between mb-5 px-1">
        <h3 className="font-heading text-[11px] font-bold uppercase tracking-widest text-slate-600 dark:text-slate-300">{label}</h3>
        <span className="text-[10px] font-mono font-semibold text-slate-400 dark:text-slate-500 bg-white dark:bg-slate-800 px-2 py-0.5 rounded border border-slate-200 dark:border-slate-700 shadow-sm">
          layer_0.kernel
        </span>
      </div>
      
      {weights ? (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-4">
          {weights.map((w, i) => (
            <div key={i} className="flex justify-between items-center py-2.5 px-4 bg-white dark:bg-slate-800/50 rounded-xl border border-slate-100 dark:border-slate-700/50 shadow-sm">
              <span className="text-[11px] font-bold text-slate-500 dark:text-slate-400 truncate max-w-[60px] md:max-w-none" title={WEIGHT_LABELS[i]}>{WEIGHT_LABELS[i]}</span>
              <span className={cn(
                "font-bold font-mono text-[11px] ml-2 transition-colors",
                w > 0 ? "text-emerald-600 dark:text-emerald-400" : "text-slate-600 dark:text-slate-400"
              )}>
                {w > 0 ? "+" : ""}{w.toFixed(4)}
              </span>
            </div>
          ))}
        </div>
      ) : (
        <div className="h-16 flex items-center justify-center text-xs text-slate-400 dark:text-slate-500 italic">
          Fetching parameters...
        </div>
      )}
    </div>
  );
}
