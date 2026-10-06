import { useState } from "react";
import {
  AlertCircle,
  BarChart3,
  CheckCircle2,
  Gauge,
  Loader2,
  Upload,
  X,
} from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const API = "http://localhost:8000";

type Analysis = {
  dataset: {
    filename: string;
    rows: number;
    columns: number;
    original_columns: string[];
    processed_features: number;
    missing_values: number;
    dropped_columns: string[];
    label_column: string | null;
    training_rows: number;
  };
  results: {
    anomalies: number;
    normal: number;
    anomaly_rate: number;
    threshold: number;
    max_score: number;
    mean_score: number;
  };
  metrics: { precision: number; recall: number; f1: number } | null;
  training: { epochs: number; final_loss: number; loss: number[] };
  error_points: { index: number; score: number; status: string }[];
  preview: Record<string, unknown>[];
};

const fmt = (v: number) =>
  new Intl.NumberFormat("en-IN", { maximumFractionDigits: 2 }).format(v);

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [drag, setDrag] = useState(false);
  const [error, setError] = useState("");

  const chooseFile = (candidate?: File) => {
    if (!candidate) return;
    if (!/\.(csv|xlsx|xls)$/i.test(candidate.name)) {
      setError("Please upload a CSV or Excel file.");
      return;
    }
    setError("");
    setFile(candidate);
    setAnalysis(null);
  };

  const runAnalysis = async () => {
    if (!file) return;
    setLoading(true);
    setError("");

    try {
      const form = new FormData();
      form.append("file", file);

      const response = await fetch(`${API}/api/analyze`, {
        method: "POST",
        body: form,
      });

      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || "Analysis failed.");
      setAnalysis(body);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not connect to the backend."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header>
        <div className="simple-title">Anomaly Detection</div>
      </header>

      <main>
        <section className="hero">
          <div>
            <em>AUTOENCODER ANALYTICS</em>
            <h1>Anomaly Detection</h1>
            <p>Upload a CSV or Excel dataset and detect unusual records.</p>
          </div>
        </section>

        <section className="workspace">
          <div className="panel upload-panel">
            <div className="heading">
              <div><small>DATASET</small><h2>Upload data</h2></div>
              {file && <span className="ready"><CheckCircle2 size={14} /> Ready</span>}
            </div>

            <div
              className={`drop ${drag ? "drag" : ""} ${file ? "selected" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
              onDragLeave={() => setDrag(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDrag(false);
                chooseFile(e.dataTransfer.files?.[0]);
              }}
            >
              <input
                id="file-input"
                hidden
                type="file"
                accept=".csv,.xlsx,.xls"
                onChange={(e) => chooseFile(e.target.files?.[0])}
              />
              <div className="upload-icon"><Upload size={20} /></div>
              <b>{file ? file.name : "Drop your dataset here"}</b>
              <span>{file ? `${(file.size / 1024 / 1024).toFixed(2)} MB` : "CSV, XLSX or XLS"}</span>
              {!file ? (
                <label htmlFor="file-input">Browse files</label>
              ) : (
                <button className="clear" type="button" onClick={() => setFile(null)}>
                  <X size={14} /> Remove
                </button>
              )}
            </div>

            <div className="upload-note">
              Labels such as <b>Class</b>, <b>Fraud</b> or <b>Anomaly</b> are detected automatically when present.
            </div>

            {error && <div className="error"><AlertCircle size={15} />{error}</div>}

            <button className="analyze" disabled={!file || loading} onClick={runAnalysis}>
              {loading ? (
                <><Loader2 size={16} className="spin" /> Analyzing dataset…</>
              ) : (
                <><Gauge size={16} /> Analyze dataset</>
              )}
            </button>
          </div>
        </section>

        {analysis && (
          <>
            <section className="result-head">
              <div>
                <em>ANALYSIS COMPLETE</em>
                <h2>{analysis.dataset.filename}</h2>
                <p>
                  {fmt(analysis.dataset.rows)} records · {analysis.dataset.columns} columns · {analysis.dataset.processed_features} features
                </p>
              </div>
            </section>

            <section className="stats">
              <Stat icon={<AlertCircle />} label="Anomalies" value={fmt(analysis.results.anomalies)} red />
              <Stat icon={<CheckCircle2 />} label="Normal records" value={fmt(analysis.results.normal)} />
              <Stat icon={<Gauge />} label="Anomaly rate" value={`${analysis.results.anomaly_rate.toFixed(2)}%`} />
              <Stat icon={<BarChart3 />} label="Threshold" value={analysis.results.threshold.toFixed(4)} />
            </section>

            <section className="dash">
              <div className="panel chart">
                <div className="heading">
                  <div><small>RECONSTRUCTION ERROR</small><h2>Anomaly score</h2></div>
                  <span>Adaptive threshold</span>
                </div>
                <div className="chartbox">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={analysis.error_points}>
                      <defs>
                        <linearGradient id="scoreFill" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#334155" stopOpacity={0.2} />
                          <stop offset="100%" stopColor="#334155" stopOpacity={0.02} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e8ecf2" vertical={false} />
                      <XAxis dataKey="index" tickLine={false} axisLine={false} tick={{ fill: "#7b8494", fontSize: 10 }} />
                      <YAxis tickLine={false} axisLine={false} tick={{ fill: "#7b8494", fontSize: 10 }} />
                      <Tooltip />
                      <ReferenceLine y={analysis.results.threshold} stroke="#b34a52" strokeDasharray="5 5" />
                      <Area type="monotone" dataKey="score" stroke="#334155" fill="url(#scoreFill)" strokeWidth={2} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="panel">
                <div className="heading">
                  <div><small>MODEL SUMMARY</small><h2>Detection profile</h2></div>
                </div>
                <div className="metrics">
                  <Metric l="Mean error" v={analysis.results.mean_score.toFixed(5)} />
                  <Metric l="Maximum score" v={analysis.results.max_score.toFixed(5)} />
                  <Metric l="Training epochs" v={String(analysis.training.epochs)} />
                  <Metric l="Training rows" v={fmt(analysis.dataset.training_rows)} />
                  <Metric l="Missing values" v={fmt(analysis.dataset.missing_values)} />
                </div>
                {analysis.metrics && (
                  <div className="evaluation">
                    <small>LABELED EVALUATION</small>
                    <div className="eval">
                      <Metric l="Precision" v={`${(analysis.metrics.precision * 100).toFixed(1)}%`} />
                      <Metric l="Recall" v={`${(analysis.metrics.recall * 100).toFixed(1)}%`} />
                      <Metric l="F1" v={`${(analysis.metrics.f1 * 100).toFixed(1)}%`} />
                    </div>
                  </div>
                )}
              </div>
            </section>

            <section className="panel table">
              <div className="heading">
                <div><small>RECORD PREVIEW</small><h2>Dataset sample</h2></div>
                <span>First 12 rows</span>
              </div>
              <div className="scroll">
                <table>
                  <thead><tr>{Object.keys(analysis.preview[0] || {}).map(k => <th key={k}>{k.replaceAll("_", " ")}</th>)}</tr></thead>
                  <tbody>
                    {analysis.preview.map((row, i) => (
                      <tr key={i}>
                        {Object.entries(row).map(([k, v]) => (
                          <td key={k}>
                            {k === "status" ? (
                              <span className={v === "Anomaly" ? "bad" : "good"}>{String(v)}</span>
                            ) : k === "anomaly_score" ? Number(v).toFixed(5) : String(v ?? "—")}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          </>
        )}

        <footer><span>Upload a dataset to begin.</span></footer>
      </main>
    </div>
  );
}

function Stat({ icon, label, value, red = false }: { icon: React.ReactNode; label: string; value: string; red?: boolean }) {
  return <div className={`stat ${red ? "red" : ""}`}><div>{icon}</div><section><span>{label}</span><b>{value}</b></section></div>;
}

function Metric({ l, v }: { l: string; v: string }) {
  return <div className="metric"><span>{l}</span><b>{v}</b></div>;
}
