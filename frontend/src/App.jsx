import { useMemo, useState } from 'react';
import {
  Activity,
  BarChart3,
  Cpu,
  Crosshair,
  MapPinned,
  Network,
  RadioTower,
  ShieldCheck
} from 'lucide-react';
import {
  architectureSteps,
  benchmarkRows,
  correctnessMetrics,
  missionBase,
  missionVariant,
  readinessCards,
  topTargets
} from './data.js';
import TacticalMap from './components/TacticalMap.jsx';
import MetricCard from './components/MetricCard.jsx';
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts';

const navItems = [
  { id: 'mission', label: 'Mission Plan', icon: MapPinned },
  { id: 'command', label: 'Tactical Command', icon: Crosshair },
  { id: 'analytics', label: 'Analytics', icon: BarChart3 },
  { id: 'architecture', label: 'System Architecture', icon: Network }
];

function Shell({ activePage, setActivePage, children }) {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">R</div>
          <div>
            <strong>ResQVision</strong>
            <span>CUDA Medical Command</span>
          </div>
        </div>
        <nav>
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                className={activePage === item.id ? 'active' : ''}
                onClick={() => setActivePage(item.id)}
              >
                <Icon size={18} />
                {item.label}
              </button>
            );
          })}
        </nav>
      </aside>
      <main>{children}</main>
    </div>
  );
}

function PageHeader({ eyebrow, title, description }) {
  return (
    <header className="page-header">
      <span>{eyebrow}</span>
      <h1>{title}</h1>
      <p>{description}</p>
    </header>
  );
}

function MissionPlan() {
  const [variant, setVariant] = useState(false);
  const mission = variant ? missionVariant : missionBase;

  return (
    <section>
      <PageHeader
        eyebrow="Pre-mission planning"
        title="Mission Plan"
        description="Configure battlefield medical coverage before the ResQVision CUDA attention engine begins live risk ranking."
      />

      <div className="mission-toolbar">
        <div>
          <h2>{mission.name}</h2>
          <p>{mission.area}</p>
        </div>
        <button className="primary-button" onClick={() => setVariant((current) => !current)}>
          Generate Mission Scenario
        </button>
      </div>

      <div className="metric-grid">
        <MetricCard label="Mission Date" value={mission.date} />
        <MetricCard label="Soldiers" value={mission.soldierCount.toLocaleString()} />
        <MetricCard label="Active ResQBands" value={mission.resqBandCount} tone="green" />
        <MetricCard label="Medical Teams" value={mission.medicalTeams} />
        <MetricCard label="UAV Available" value={mission.uavAvailable ? 'Online' : 'Offline'} tone="green" />
        <MetricCard label="LoRa Relays" value={mission.loraRelays} />
        <MetricCard label="EVAC Zones" value={mission.evacuationZones} tone="orange" />
        <MetricCard label="Readiness Score" value={`${mission.readinessScore}%`} tone="green" />
      </div>

      <div className="content-grid two-columns">
        <section className="panel map-panel">
          <div className="panel-title">
            <h3>Planning Map</h3>
            <span>1000 x 1000 tactical grid</span>
          </div>
          <TacticalMap planning />
        </section>
        <section className="readiness-stack">
          {readinessCards.map((card) => (
            <article className="panel readiness-card" key={card.title}>
              <ShieldCheck size={22} />
              <div>
                <h3>{card.title}</h3>
                <strong>{card.value}</strong>
                <p>{card.detail}</p>
              </div>
            </article>
          ))}
        </section>
      </div>
    </section>
  );
}

function TacticalCommand() {
  return (
    <section>
      <PageHeader
        eyebrow="Live tactical command"
        title="Tactical Command"
        description="Real-time casualty prioritization with UAV routing, ResQBand telemetry, and CUDA-accelerated attention scores."
      />

      <div className="command-layout">
        <section className="panel map-panel">
          <div className="panel-title">
            <h3>Live Battlefield Map</h3>
            <div className="legend">
              <span><i className="dot critical" /> Critical</span>
              <span><i className="dot urgent" /> Urgent</span>
              <span><i className="dot stable" /> Stable</span>
            </div>
          </div>
          <TacticalMap showArrows />
        </section>

        <aside className="panel targets-panel">
          <div className="engine-card">
            <Cpu />
            <div>
              <span>CUDA Engine</span>
              <strong>Online</strong>
            </div>
          </div>
          <div className="status-row">
            <span>Correctness</span>
            <strong className="pass">PASS</strong>
          </div>
          <div className="status-row">
            <span>Top-10 overlap</span>
            <strong>10/10</strong>
          </div>
          <h3>Top 10 Evacuation Targets</h3>
          <div className="target-list">
            {topTargets.map((target) => (
              <article key={target.id} className={`target-card ${target.category}`}>
                <b>#{target.rank}</b>
                <div>
                  <strong>{target.id} · {target.name}</strong>
                  <span>HR {target.hr} bpm · SpO2 {target.spo2}%</span>
                </div>
                <em>{Math.round(target.risk * 100)}%</em>
              </article>
            ))}
          </div>
        </aside>
      </div>
    </section>
  );
}

function Analytics() {
  const chartData = useMemo(
    () => benchmarkRows.map((row) => ({ soldiers: row.soldiers, CPU: row.cpu, GPU: row.gpu, speedup: row.speedup })),
    []
  );

  return (
    <section>
      <PageHeader
        eyebrow="Computational results"
        title="Analytics"
        description="Benchmark and correctness signals from the ResQVision CUDA attention engine, presented for static demonstration."
      />

      <div className="content-grid two-columns">
        <section className="panel chart-panel">
          <div className="panel-title">
            <h3>CPU vs GPU Runtime</h3>
            <span>Milliseconds by scenario size</span>
          </div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#dbe7f5" />
              <XAxis dataKey="soldiers" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="CPU" fill="#93c5fd" radius={[6, 6, 0, 0]} />
              <Bar dataKey="GPU" fill="#2563eb" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </section>
        <section className="panel chart-panel">
          <div className="panel-title">
            <h3>GPU Speedup</h3>
            <span>Attention kernel acceleration</span>
          </div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#dbe7f5" />
              <XAxis dataKey="soldiers" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="speedup" fill="#16a34a" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </section>
      </div>

      <div className="content-grid two-columns">
        <section className="panel">
          <div className="panel-title">
            <h3>Benchmark Table</h3>
            <span>Hardcoded sample output</span>
          </div>
          <table>
            <thead>
              <tr>
                <th>Soldiers</th>
                <th>CPU ms</th>
                <th>GPU ms</th>
                <th>Speedup</th>
              </tr>
            </thead>
            <tbody>
              {benchmarkRows.map((row) => (
                <tr key={row.soldiers}>
                  <td>{row.soldiers}</td>
                  <td>{row.cpu}</td>
                  <td>{row.gpu}</td>
                  <td>{row.speedup}x</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
        <section className="panel metrics-panel">
          <h3>Correctness Metrics</h3>
          <div className="status-row">
            <span>Status</span>
            <strong className="pass">{correctnessMetrics.status}</strong>
          </div>
          <div className="status-row">
            <span>Top-10 overlap</span>
            <strong>{correctnessMetrics.top10Overlap}</strong>
          </div>
          <div className="status-row">
            <span>Max abs error</span>
            <strong>{correctnessMetrics.maxAbsError}</strong>
          </div>
          <div className="status-row">
            <span>Mean abs error</span>
            <strong>{correctnessMetrics.meanAbsError}</strong>
          </div>
          <p>{correctnessMetrics.attentionEntropy}</p>
          <p>GPU acceleration enables ResQVision to scale attention-based risk assessment toward larger battlefield scenarios while preserving CPU/GPU correctness validation.</p>
        </section>
      </div>
    </section>
  );
}

function SystemArchitecture() {
  return (
    <section>
      <PageHeader
        eyebrow="Pipeline architecture"
        title="System Architecture"
        description="A static overview of how visual detection, wearable telemetry, CUDA attention, and command visualization fit together."
      />

      <section className="panel pipeline-panel">
        <h3>Operational Pipeline</h3>
        <div className="pipeline">
          {architectureSteps.map((step, index) => (
            <div className="pipeline-step" key={step}>
              <span>{index + 1}</span>
              <strong>{step}</strong>
            </div>
          ))}
        </div>
      </section>

      <div className="content-grid two-columns">
        <section className="panel cuda-panel">
          <Cpu size={28} />
          <h3>CUDA Attention Engine</h3>
          <ul>
            <li>QK^T matrix multiplication computes soldier-to-soldier attention affinity.</li>
            <li>Scaling stabilizes the attention logits before normalization.</li>
            <li>Row-wise softmax produces probability distributions over battlefield context.</li>
            <li>Attention @ V aggregates risk-relevant telemetry and visual features.</li>
            <li>CPU/GPU correctness validation confirms numerical agreement and top-rank overlap.</li>
          </ul>
        </section>
        <section className="panel architecture-note">
          <RadioTower size={28} />
          <h3>Defense-Medical Data Flow</h3>
          <p>
            ResQVision fuses UAV observation, YOLO soldier localization, and ResQBand telemetry into a feature matrix that the CUDA kernel can rank at operational speed.
          </p>
          <p>
            The tactical dashboard turns those rankings into evacuation targets, routing cues, and medical command readiness indicators.
          </p>
        </section>
      </div>
    </section>
  );
}

export default function App() {
  const [activePage, setActivePage] = useState('mission');

  const page = {
    mission: <MissionPlan />,
    command: <TacticalCommand />,
    analytics: <Analytics />,
    architecture: <SystemArchitecture />
  }[activePage];

  return (
    <Shell activePage={activePage} setActivePage={setActivePage}>
      <div className="top-strip">
        <div>
          <Activity size={18} />
          Static research demo · no backend · sample CUDA benchmark data
        </div>
        <span>Operational Readiness: 94%</span>
      </div>
      {page}
    </Shell>
  );
}
