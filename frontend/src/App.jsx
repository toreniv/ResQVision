import { useEffect, useMemo, useState } from 'react';
import {
  Activity,
  BarChart3,
  Clock,
  Cpu,
  Crosshair,
  MapPinned,
  Network,
  Plane,
  RadioTower,
  ShieldCheck
} from 'lucide-react';
import {
  architectureSteps,
  benchmarkRows,
  correctnessMetrics,
  missionBase,
  missionVariant,
  topTargets
} from './data.js';
import TacticalMap from './components/TacticalMap.jsx';
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
    <div className={`app-shell ${activePage === 'mission' ? 'mission-active' : ''} ${activePage === 'command' ? 'command-active' : ''}`}>
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
  const [currentTime, setCurrentTime] = useState(() => new Date());
  const mission = variant ? missionVariant : missionBase;
  const criticalCount = topTargets.filter((target) => target.category === 'critical').length;
  const urgentCount = topTargets.filter((target) => target.category === 'urgent').length;
  const stableCount = Math.max(0, mission.soldierCount - criticalCount - urgentCount);

  useEffect(() => {
    const timer = window.setInterval(() => setCurrentTime(new Date()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  const utcTime = currentTime.toLocaleTimeString('en-US', {
    hour12: false,
    timeZone: 'UTC'
  });
  const utcDate = currentTime.toLocaleDateString('en-US', {
    day: '2-digit',
    month: 'short',
    timeZone: 'UTC',
    year: 'numeric'
  });

  return (
    <section className="mission-console">
      <header className="tactical-command-header">
        <div className="command-brand-card">
          <div className="command-shield">R</div>
          <div>
            <strong>RESQVISION</strong>
            <span>AI for battlefield medicine</span>
          </div>
        </div>

        <div className="command-title">
          <h1><span>RESQ</span>VISION TACTICAL COMMAND</h1>
          <p>GPU-Accelerated Battlefield Casualty Prioritization | CUDA Attention Engine</p>
          <small>Static demo mode - ready for CUDA JSON integration</small>
        </div>

        <div className="command-header-cards">
          <article className="header-status-card">
            <Plane size={38} />
            <div>
              <span>UAV-1 Status</span>
              <strong>Operational</strong>
            </div>
            <i className="status-light" />
          </article>
          <article className="header-status-card time-card">
            <Clock size={22} />
            <div>
              <span>Time (UTC)</span>
              <strong>{utcTime}</strong>
              <small>{utcDate}</small>
            </div>
          </article>
          <article className="header-status-card mission-live-card">
            <ShieldCheck size={24} />
            <div>
              <span>Mission Status</span>
              <strong>Active</strong>
            </div>
          </article>
        </div>
      </header>

      <div className="live-command-grid">
        <section className="command-map-surface">
          <TacticalMap planning showArrows />
        </section>

        <aside className="evac-targets-console">
          <div className="targets-console-title">
            <Crosshair size={24} />
            <h2>Top Evacuation Targets</h2>
          </div>
          <div className="mission-meta-line">
            <span>{mission.name}</span>
            <button onClick={() => setVariant((current) => !current)}>Generate Scenario</button>
          </div>

          <div className="evac-target-list">
            {topTargets.map((target) => (
              <article className={`evac-target-row severity-${target.category}`} key={target.id}>
                <div className="rank-badge">{target.rank}</div>
                <div className="evac-target-body">
                  <div className="target-row-top">
                    <strong>Soldier ID: {target.id}</strong>
                    <em>{(target.risk * 100).toFixed(1)}</em>
                  </div>
                  <div className="target-vitals">
                    <span>HR: {target.hr} bpm</span>
                    <span>SpO2: {target.spo2}%</span>
                  </div>
                  <div className="risk-track">
                    <i style={{ width: `${Math.round(target.risk * 100)}%` }} />
                  </div>
                </div>
              </article>
            ))}
          </div>

          <div className="console-legend">
            <span><i className="dot critical" /> Critical</span>
            <span><i className="dot urgent" /> Urgent</span>
            <span><i className="dot stable" /> Stable</span>
            <span><i className="marker-sample uav" /> UAV-1</span>
          </div>
        </aside>
      </div>

      <footer className="operation-stat-strip">
        <div><i className="strip-icon critical-icon" /><span>Critical</span><strong className="critical-text">{criticalCount}</strong></div>
        <div><i className="strip-icon urgent-icon" /><span>Urgent</span><strong className="urgent-text">{urgentCount}</strong></div>
        <div><i className="strip-icon stable-icon" /><span>Stable</span><strong className="stable-text">{stableCount}</strong></div>
        <div><i className="strip-icon uav-icon" /><span>UAV Location</span><strong>(510 m, 130 m)</strong></div>
        <div><i className="strip-icon cuda-icon" /><span>CUDA Engine</span><strong>Attention PASS</strong></div>
        <div><i className="strip-icon data-icon" /><span>Data Source</span><strong>ResQBand Telemetry</strong></div>
        <div><i className="strip-icon mission-icon" /><span>Mission Status</span><strong className="stable-text">Active</strong></div>
      </footer>
    </section>
  );
}

function TacticalCommand({ setActivePage }) {
  const [currentTime, setCurrentTime] = useState(() => new Date());
  const criticalCount = topTargets.filter((target) => target.category === 'critical').length;
  const urgentCount = topTargets.filter((target) => target.category === 'urgent').length;
  const stableCount = Math.max(0, missionBase.soldierCount - criticalCount - urgentCount);

  useEffect(() => {
    const timer = window.setInterval(() => setCurrentTime(new Date()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  const utcTime = currentTime.toLocaleTimeString('en-US', {
    hour12: false,
    timeZone: 'UTC'
  });

  return (
    <section className="tactical-live-console">
      <header className="ops-header">
        <div className="ops-logo">
          <div className="command-shield">R</div>
          <div>
            <strong>RESQVISION</strong>
            <span>AI for battlefield medicine</span>
          </div>
        </div>

        <div className="ops-title-block">
          <h1><span>RESQ</span>VISION TACTICAL COMMAND</h1>
          <p>GPU-Accelerated Battlefield Casualty Prioritization | CUDA Attention Engine</p>
          <nav className="ops-top-nav" aria-label="Primary navigation">
            {navItems.map((item) => (
              <button
                key={item.id}
                className={item.id === 'command' ? 'active' : ''}
                onClick={() => setActivePage(item.id)}
              >
                {item.label}
              </button>
            ))}
          </nav>
        </div>

        <div className="ops-status-cluster">
          <article>
            <Plane size={28} />
            <span>UAV-1 Status</span>
            <strong>Operational</strong>
          </article>
          <article>
            <Clock size={22} />
            <span>Time (UTC)</span>
            <strong>{utcTime}</strong>
          </article>
          <article>
            <ShieldCheck size={24} />
            <span>Mission Status</span>
            <strong>Active</strong>
          </article>
        </div>
      </header>

      <div className="ops-main-grid">
        <section className="ops-map-panel">
          <TacticalMap planning showArrows />
        </section>

        <aside className="live-evac-panel">
          <div className="live-panel-title">
            <Crosshair size={22} />
            <h2>Top Evacuation Targets</h2>
          </div>
          <div className="live-panel-subtitle">
            <span>Risk Score</span>
            <span>Top-10 overlap 10/10</span>
          </div>

          <div className="live-target-list">
            {topTargets.slice(0, 10).map((target) => (
              <article className={`live-target-row severity-${target.category}`} key={target.id}>
                <div className="live-rank">{target.rank}</div>
                <div className="live-target-content">
                  <div className="live-target-top">
                    <strong>Soldier ID: {target.id}</strong>
                    <em>{(target.risk * 100).toFixed(1)}</em>
                  </div>
                  <div className="live-vitals">
                    <span>HR: {target.hr} bpm</span>
                    <span>SpO2: {target.spo2}%</span>
                  </div>
                  <div className="live-risk-bar">
                    <i style={{ width: `${Math.round(target.risk * 100)}%` }} />
                  </div>
                </div>
              </article>
            ))}
          </div>
        </aside>
      </div>

      <footer className="live-status-bar">
        <div><span>Critical</span><strong className="critical-text">{criticalCount}</strong></div>
        <div><span>Urgent</span><strong className="urgent-text">{urgentCount}</strong></div>
        <div><span>Stable</span><strong className="stable-text">{stableCount}</strong></div>
        <div><span>UAV Location</span><strong>(510 m, 130 m)</strong></div>
        <div><span>CUDA Engine</span><strong>Attention PASS</strong></div>
        <div><span>Data Source</span><strong>ResQBand Telemetry</strong></div>
        <div><span>Mission Status</span><strong className="stable-text">Active</strong></div>
      </footer>
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
    command: <TacticalCommand setActivePage={setActivePage} />,
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
