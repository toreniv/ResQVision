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
  readinessCards,
  topTargets
} from './data.js';
import { loadAttentionStats, loadBenchmarkResults, loadRiskRanking } from './dataLoader.js';
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

function useCudaData() {
  const [benchmarks, setBenchmarks] = useState(null);
  const [riskRanking, setRiskRanking] = useState(null);
  const [attentionStats, setAttentionStats] = useState(null);

  useEffect(() => {
    loadBenchmarkResults().then((data) => data && setBenchmarks(data));
    loadRiskRanking().then((data) => data && setRiskRanking(data));
    loadAttentionStats().then((data) => data && setAttentionStats(data));
  }, []);

  return { benchmarks, riskRanking, attentionStats };
}

function Shell({ activePage, setActivePage, children }) {
  return (
    <div className={`app-shell ${activePage === 'command' ? 'command-active' : ''}`}>
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

function MissionPlan({ riskRanking, attentionStats, setActivePage, setExpandedMap }) {
  const [variant, setVariant] = useState(false);
  const mission = variant ? missionVariant : missionBase;
  const liveSoldiers = riskRanking ?? topTargets;
  const criticalCount = liveSoldiers.filter((target) => target.category === 'critical').length;
  const urgentCount = liveSoldiers.filter((target) => target.category === 'urgent').length;
  const stableCount = Math.max(0, mission.soldierCount - criticalCount - urgentCount);
  const missionChecklist = ['Identify casualties', 'Validate telemetry', 'Prepare corridors', 'Confirm CUDA PASS'];
  const missionTimeline = [
    'Scenario loaded',
    'Telemetry validated',
    'CUDA baseline passed',
    'UAV route planned',
    'Ready for command'
  ];

  return (
    <section className="mission-page">
      <div className="mission-header-row">
        <PageHeader
          eyebrow="Mission planning"
          title="Mission Plan"
          description="Pre-operation overview before entering Tactical Command."
        />
        <div className="mission-actions">
          <button className="secondary-button compact-button" onClick={() => setVariant((current) => !current)}>
            Generate Scenario
          </button>
          <button className="primary-button compact-button" onClick={() => setActivePage('command')}>
            Enter Tactical Command
          </button>
        </div>
      </div>

      <div className="mission-summary-grid">
        <article className="metric-card metric-blue">
          <span>Mission name</span>
          <strong>{mission.name}</strong>
          <small>Scenario profile</small>
        </article>
        <article className="metric-card">
          <span>Area</span>
          <strong>{mission.area}</strong>
          <small>Planning sector</small>
        </article>
        <article className="metric-card metric-green">
          <span>Soldiers tracked</span>
          <strong>{mission.soldierCount.toLocaleString()}</strong>
          <small>{mission.resqBandCount} ResQBands online</small>
        </article>
        <article className="metric-card metric-orange">
          <span>Readiness score</span>
          <strong>{mission.readinessScore}%</strong>
          <small>{mission.medicalTeams} teams / {mission.evacuationZones} EVAC zones</small>
        </article>
      </div>

      <div className="mission-plan-main">
        <section className="panel mission-map-panel">
          <div className="map-panel-header">
            <div>
              <h3>Mission Area</h3>
              <span>Planned UAV route and evacuation corridors</span>
            </div>
            <button className="map-expand-button" onClick={() => setExpandedMap('mission')}>
              Expand Map
            </button>
          </div>
          <TacticalMap planning showArrows soldiers={liveSoldiers} attentionData={attentionStats ?? []} />
        </section>

        <aside className="mission-briefing-stack">
          <section className="panel mission-compact-panel">
            <div className="panel-title">
              <h3>Readiness</h3>
              <span>Pre-op</span>
            </div>
            <div className="readiness-grid">
              {readinessCards.map((card) => (
                <article className="readiness-card" key={card.title}>
                  <span>{card.title.replace(' Readiness', '')}</span>
                  <strong>{card.value}</strong>
                </article>
              ))}
            </div>
          </section>

          <section className="panel mission-compact-panel">
            <div className="panel-title">
              <h3>Briefing Checklist</h3>
              <span>Required</span>
            </div>
            <div className="checklist-chip-grid">
              {missionChecklist.map((item) => (
                <span key={item}>{item}</span>
              ))}
            </div>
          </section>

          <section className="panel mission-compact-panel">
            <div className="panel-title">
              <h3>Field Assets</h3>
              <span>Available</span>
            </div>
            <div className="asset-chip-grid">
              <span>UAV available</span>
              <span>{mission.medicalTeams} medical teams</span>
              <span>{mission.loraRelays} LoRa relays</span>
              <span>{mission.resqBandCount} ResQBands</span>
              <span>{mission.evacuationZones} EVAC zones</span>
            </div>
          </section>

          <section className="panel mission-compact-panel">
            <div className="panel-title">
              <h3>Mission Timeline</h3>
              <span>Sequence</span>
            </div>
            <div className="compact-timeline">
              {missionTimeline.map((step, index) => (
                <div className="compact-timeline-step" key={step}>
                  <i>{index + 1}</i>
                  <span>{step}</span>
                </div>
              ))}
            </div>
          </section>

          <section className="panel mission-compact-panel">
            <div className="panel-title">
              <h3>Risk Summary</h3>
              <span>Preview</span>
            </div>
            <div className="risk-summary-list compact-risk-summary">
              <div><span>Critical</span><strong className="critical-text">{criticalCount}</strong></div>
              <div><span>Urgent</span><strong className="urgent-text">{urgentCount}</strong></div>
              <div><span>Stable</span><strong className="stable-text">{stableCount}</strong></div>
              <div><span>Top cluster</span><strong>S-0412 / S-0188 / S-0774</strong></div>
            </div>
          </section>

          <section className="panel mission-compact-panel mission-action-panel">
            <button className="primary-button compact-button" onClick={() => setActivePage('command')}>
              Enter Tactical Command
            </button>
          </section>
        </aside>
      </div>
    </section>
  );
}

function deriveRecommendedActions(targets) {
  if (!targets || targets.length === 0) return [];

  const critical = targets.filter(t => t.category === 'critical');
  const urgent   = targets.filter(t => t.category === 'urgent');
  const top      = targets[0];
  const second   = targets[1];

  const actions = [];

  // Action 1 – always present
  actions.push({
    priority: 1,
    title: `Evacuate Soldier ${top.id}`,
    reason: `Risk ${(top.risk * 100).toFixed(1)} · HR ${top.hr} bpm · SpO2 ${top.spo2}%`,
    level: 'critical',
  });

  // Action 2 – trauma team if 2+ critical, otherwise stabilize second
  if (critical.length >= 2) {
    actions.push({
      priority: 2,
      title: 'Dispatch Trauma Team Bravo',
      reason: `${critical.length} critical casualties in sector`,
      level: 'critical',
    });
  } else if (second) {
    actions.push({
      priority: 2,
      title: `Stabilize Soldier ${second.id}`,
      reason: `Risk ${(second.risk * 100).toFixed(1)} · monitor vitals`,
      level: second.category,
    });
  }

  // Action 3 – UAV routing always present
  actions.push({
    priority: 3,
    title: 'Route UAV-1 to Casualty Cluster',
    reason: `Top ${Math.min(3, targets.length)} targets within operational range`,
    level: 'urgent',
  });

  // Action 4 – highest urgent non-critical, or 4th soldier
  const monitorTarget = urgent[0] ?? targets[3];
  if (monitorTarget) {
    actions.push({
      priority: 4,
      title: `Monitor Soldier ${monitorTarget.id}`,
      reason: `HR ${monitorTarget.hr} bpm · SpO2 ${monitorTarget.spo2}% · trend watch`,
      level: 'stable',
    });
  }

  return actions;
}

function TacticalCommand({ riskRanking, attentionStats, setExpandedMap }) {
  const [currentTime, setCurrentTime] = useState(() => new Date());
  const liveSoldiers = riskRanking ?? topTargets;
  const criticalCount = liveSoldiers.filter((target) => target.category === 'critical').length;
  const urgentCount = liveSoldiers.filter((target) => target.category === 'urgent').length;
  const stableCount = Math.max(0, missionBase.soldierCount - criticalCount - urgentCount);
  const visibleTargets = liveSoldiers.slice(0, 5);
  const recommendedActions = deriveRecommendedActions(visibleTargets);

  useEffect(() => {
    const timer = window.setInterval(() => setCurrentTime(new Date()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  const utcTime = currentTime.toLocaleTimeString('en-US', {
    hour12: false,
    timeZone: 'UTC'
  });

  return (
    <section className="tc-page">
      <header className="tc-header">
        <div className="tc-header-left">
          <div className="tc-brand">
            <div className="tc-shield">R</div>
            <div>
              <strong>RESQVISION</strong>
              <span>AI for battlefield medicine</span>
            </div>
          </div>

          <div>
            <h1><span>RESQ</span>VISION TACTICAL COMMAND</h1>
            <p>GPU-Accelerated Battlefield Casualty Prioritization | CUDA Attention Engine</p>
          </div>
        </div>

        <div className="tc-header-right">
          <article>
            <Plane size={16} />
            <span>UAV-1 Status</span>
            <strong>Operational</strong>
          </article>
          <article>
            <Clock size={16} />
            <span>Time (UTC)</span>
            <strong>{utcTime}</strong>
          </article>
          <article>
            <ShieldCheck size={16} />
            <span>Mission Status</span>
            <strong>Active</strong>
          </article>
        </div>
      </header>

      <div className="tc-main">
        <section className="tc-map-panel">
          <div className="map-panel-header tc-map-header">
            <div>
              <h3>Tactical Map</h3>
              <span>Live evacuation command view</span>
            </div>
            <button className="map-expand-button" onClick={() => setExpandedMap('command')}>
              Expand Map
            </button>
          </div>
          <TacticalMap planning showArrows soldiers={liveSoldiers} attentionData={attentionStats ?? []} />
        </section>

        <aside className="tc-side-panel">
          <div className="tc-panel-title">
            <Crosshair size={18} />
            <h2>Top Evacuation Targets</h2>
          </div>
          <p>Sorted by risk score</p>

          <div className="tc-target-list">
            {visibleTargets.map((target) => (
              <article className={`tc-target-row tc-severity-${target.category}`} key={target.id}>
                <div className="tc-target-rank">{target.rank}</div>
                <div className="tc-target-body">
                  <div className="tc-target-top">
                    <strong>Soldier ID: {target.id}</strong>
                    <em>{(target.risk * 100).toFixed(1)}</em>
                  </div>
                  <div className="tc-target-vitals">
                    <span>HR: {target.hr} bpm</span>
                    <span>SpO2: {target.spo2}%</span>
                  </div>
                  <div className="tc-target-bar">
                    <i style={{ width: `${Math.round(target.risk * 100)}%` }} />
                  </div>
                </div>
              </article>
            ))}
          </div>

          <div className="recommended-actions">
            <div className="recommended-actions-title">
              <ShieldCheck size={14} />
              <span>Recommended Actions</span>
            </div>
            {recommendedActions.map(action => (
              <div className={`action-row action-${action.level}`} key={action.priority}>
                <div className="action-priority">{action.priority}</div>
                <div className="action-body">
                  <strong>{action.title}</strong>
                  <span>{action.reason}</span>
                </div>
              </div>
            ))}
          </div>

          <div className="tc-legend">
            <span><i className="tc-dot tc-critical" /> Critical</span>
            <span><i className="tc-dot tc-urgent" /> Urgent</span>
            <span><i className="tc-dot tc-stable" /> Stable</span>
            <span><i className="tc-marker-sample tc-uav" /> UAV-1</span>
          </div>
        </aside>
      </div>

      <footer className="tc-status-bar">
        <div><Crosshair size={14} /><span>Critical</span><strong className="tc-critical-text">{criticalCount}</strong></div>
        <div><Activity size={14} /><span>Urgent</span><strong className="tc-urgent-text">{urgentCount}</strong></div>
        <div><ShieldCheck size={14} /><span>Stable</span><strong className="tc-stable-text">{stableCount}</strong></div>
        <div><Cpu size={14} /><span>CUDA</span><strong>Attention PASS</strong></div>
        <div><RadioTower size={14} /><span>Data</span><strong>ResQBand</strong></div>
        <div><ShieldCheck size={14} /><span>Mission</span><strong className="tc-stable-text">Active</strong></div>
      </footer>
    </section>
  );
}

function Analytics({ benchmarks, attentionStats }) {
  const benchmarkSource = benchmarks ?? benchmarkRows;
  const chartData = useMemo(
    () => {
      return benchmarkSource.map((row) => ({
        soldiers: row.soldiers,
        CPU: row.cpu ?? row.CPU,
        GPU: row.gpu ?? row.GPU,
        speedup: row.speedup
      }));
    },
    [benchmarkSource]
  );
  const lastBenchmark = benchmarks?.length
    ? benchmarks.reduce((largest, row) => (row.soldiers > largest.soldiers ? row : largest), benchmarks[0])
    : null;
  const metrics = lastBenchmark
    ? {
        status: lastBenchmark.correctness ?? 'PASS',
        top10Overlap: `${lastBenchmark.top10Overlap}/10`,
        maxAbsError: lastBenchmark.maxAbsError,
        meanAbsError: lastBenchmark.meanAbsError,
        attentionEntropy: correctnessMetrics.attentionEntropy
      }
    : attentionStats ?? correctnessMetrics;

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
            <span>{benchmarks ? 'CUDA benchmark output' : 'Mock fallback output'}</span>
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
              {benchmarkSource.map((row) => (
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
            <strong className="pass">{metrics.status}</strong>
          </div>
          <div className="status-row">
            <span>Top-10 overlap</span>
            <strong>{metrics.top10Overlap}</strong>
          </div>
          <div className="status-row">
            <span>Max abs error</span>
            <strong>{metrics.maxAbsError}</strong>
          </div>
          <div className="status-row">
            <span>Mean abs error</span>
            <strong>{metrics.meanAbsError}</strong>
          </div>
          <p>{metrics.attentionEntropy ?? correctnessMetrics.attentionEntropy}</p>
          <p>GPU acceleration enables ResQVision to scale attention-based risk assessment toward larger battlefield scenarios while preserving CPU/GPU correctness validation.</p>
        </section>
      </div>
    </section>
  );
}

function SystemArchitecture() {
  const realPipelineSteps = [
    'CUDA / Colab Simulation',
    'CSV Artifact Export',
    'JSON Conversion',
    'React dataLoader',
    'Benchmark Analytics',
    'Risk Ranking',
    'Tactical Map Markers',
    'Evacuation Decision Queue'
  ];

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
          {realPipelineSteps.map((step, index) => (
            <div className="pipeline-step" key={step}>
              <span>{index + 1}</span>
              <strong>{step}</strong>
            </div>
          ))}
        </div>
      </section>

      <div className="content-grid two-columns">
        <section className="panel cuda-panel">
          <Network size={28} />
          <h3>Live Data Flow</h3>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap', marginTop: '18px' }}>
            <div className="pipeline-step" style={{ minHeight: 'auto', padding: '12px', flex: '1 1 auto' }}><strong>CUDA C++ Attention Engine</strong></div>
            <span style={{ color: 'var(--muted)', fontWeight: 'bold' }}>→</span>
            <div className="pipeline-step" style={{ minHeight: 'auto', padding: '12px', flex: '1 1 auto' }}><strong>benchmark_results.json</strong></div>
            <span style={{ color: 'var(--muted)', fontWeight: 'bold' }}>→</span>
            <div className="pipeline-step" style={{ minHeight: 'auto', padding: '12px', flex: '1 1 auto' }}><strong>risk_ranking.json</strong></div>
            <span style={{ color: 'var(--muted)', fontWeight: 'bold' }}>→</span>
            <div className="pipeline-step" style={{ minHeight: 'auto', padding: '12px', flex: '1 1 auto' }}><strong>attention_stats.json</strong></div>
            <span style={{ color: 'var(--muted)', fontWeight: 'bold' }}>→</span>
            <div className="pipeline-step" style={{ minHeight: 'auto', padding: '12px', flex: '1 1 auto' }}><strong>React dashboard</strong></div>
          </div>
        </section>
        <section className="panel cuda-panel">
          <ShieldCheck size={28} />
          <h3>Fallback Safety Layer</h3>
          <ul>
            <li>If JSON files exist, dashboard uses CUDA output.</li>
            <li>If JSON files are missing or malformed, dashboard falls back to mock data.</li>
            <li>This keeps the demo stable.</li>
          </ul>
        </section>
      </div>

      <div className="content-grid two-columns">
        <section className="panel cuda-panel">
          <Activity size={28} />
          <h3>What is still simulated</h3>
          <ul>
            <li>UAV imagery</li>
            <li>YOLO detection</li>
            <li>ResQBand live telemetry</li>
            <li>Real-time backend</li>
            <li>Clinical decision approval</li>
          </ul>
        </section>
        <section className="panel cuda-panel">
          <Cpu size={28} />
          <h3>What is real now</h3>
          <ul>
            <li>CUDA benchmark output</li>
            <li>Risk ranking JSON</li>
            <li>Attention statistics JSON</li>
            <li>Analytics charts</li>
            <li>Tactical Command top targets</li>
            <li>Tactical map casualty markers</li>
          </ul>
        </section>
      </div>
    </section>
  );
}

export default function App() {
  const [activePage, setActivePage] = useState('mission');
  const [expandedMap, setExpandedMap] = useState(null);
  const cudaData = useCudaData();

  useEffect(() => {
    if (!expandedMap) {
      return undefined;
    }

    const handleEscape = (event) => {
      if (event.key === 'Escape') {
        setExpandedMap(null);
      }
    };

    window.addEventListener('keydown', handleEscape);
    return () => window.removeEventListener('keydown', handleEscape);
  }, [expandedMap]);

  const page = {
    mission: <MissionPlan riskRanking={cudaData.riskRanking} attentionStats={cudaData.attentionStats} setActivePage={setActivePage} setExpandedMap={setExpandedMap} />,
    command: <TacticalCommand riskRanking={cudaData.riskRanking} attentionStats={cudaData.attentionStats} setExpandedMap={setExpandedMap} />,
    analytics: <Analytics benchmarks={cudaData.benchmarks} attentionStats={cudaData.attentionStats} />,
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
      {expandedMap ? (
        <div className="map-modal-backdrop" role="dialog" aria-modal="true" aria-label={expandedMap === 'mission' ? 'Mission Area Map' : 'Tactical Command Map'}>
          <section className="map-modal">
            <header className="map-modal-header">
              <h2>{expandedMap === 'mission' ? 'Mission Area Map' : 'Tactical Command Map'}</h2>
              <button onClick={() => setExpandedMap(null)}>Close</button>
            </header>
            <div className="map-modal-body">
              <TacticalMap planning showArrows />
            </div>
          </section>
        </div>
      ) : null}
    </Shell>
  );
}
