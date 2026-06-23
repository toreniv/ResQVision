import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  BarChart3,
  Camera,
  Clock,
  Cpu,
  Crosshair,
  Eye,
  MapPinned,
  Network,
  Plane,
  RadioTower,
  ShieldCheck,
  Upload
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
import BrowserCvDetector from './components/BrowserCvDetector.jsx';
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
  { id: 'architecture', label: 'System Architecture', icon: Network },
  { id: 'cv', label: 'Computer Vision', icon: Eye }
];

function useCudaData() {
  const [benchmarks, setBenchmarks] = useState(null);
  const [riskRanking, setRiskRanking] = useState(null);
  const [attentionStats, setAttentionStats] = useState(null);
  const [fusionMode, setFusionMode] = useState(null);

  const refresh = () => {
    loadBenchmarkResults().then((data) => data && setBenchmarks(data));
    loadRiskRanking().then((data) => {
      if (data) {
        setFusionMode(data.fusionMode ?? null);
        setRiskRanking(data);
      }
    });
    loadAttentionStats().then((data) => data && setAttentionStats(data));
  };

  useEffect(() => {
    refresh();
  }, []);

  return { benchmarks, riskRanking, attentionStats, fusionMode, refresh };
}

function Shell({ activePage, setActivePage, children }) {
  return (
    <div className={`app-shell ${activePage === 'command' ? 'command-active' : ''} ${activePage === 'mission' ? 'mission-active' : ''}`}>
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

function PageExplainer({ children, tags = [] }) {
  return (
    <div className="page-explainer">
      <strong>What this page shows</strong>
      <p>{children}</p>
      {tags.length ? (
        <div className="page-tags">
          {tags.map((tag) => (
            <span key={tag}>{tag}</span>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function MissionPlan({ riskRanking, attentionStats, fusionMode, manualDronePoints, setActivePage, setExpandedMap }) {
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
      <PageExplainer tags={['Demo / Simulated', 'Local GPU Mode', 'Colab Fallback']}>
        Pre-operation scenario view. Tactical Command uses the imported CUDA risk data after the scenario is loaded.
      </PageExplainer>

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
          <TacticalMap planning showArrows soldiers={liveSoldiers} attentionData={attentionStats ?? []} fusionMode={fusionMode} manualPoints={manualDronePoints} />
        </section>

        <aside className="mission-briefing-stack">
          <section className="panel mission-compact-panel">
            <div className="panel-title">
              <h3>Readiness</h3>
              <span>Readiness status</span>
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
              <span>Pre-op summary</span>
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
              <span>Mission context</span>
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
              <span>Next action</span>
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
              <p>Full ranking in Tactical Command.</p>
            </div>
          </section>
        </aside>
      </div>
    </section>
  );
}

function deriveRecommendedActions(targets) {
  if (!targets || targets.length === 0) return [];

  if (targets.some(target => target.recommendedAction)) {
    return targets.slice(0, 4).map((target, index) => ({
      priority: index + 1,
      title: target.recommendedAction,
      reason: `${target.id} - Risk ${Number.isFinite(target.risk) ? (target.risk * 100).toFixed(1) : 'UNLINKED'}${target.confidence ? ` - YOLO ${(target.confidence * 100).toFixed(1)}%` : ''}`,
      level: target.category,
    }));
  }

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

function TacticalCommand({ riskRanking, attentionStats, fusionMode, manualDronePoints, setExpandedMap }) {
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
      <PageExplainer tags={['Real / Implemented', 'Demo / Simulated']}>
        Uses CUDA-generated risk ranking and attention statistics to prioritize evacuation targets.
      </PageExplainer>

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
          <TacticalMap planning showArrows soldiers={liveSoldiers} attentionData={attentionStats ?? []} fusionMode={fusionMode} manualPoints={manualDronePoints} />
        </section>

        <aside className="tc-side-panel">
          <div className="tc-panel-title">
            <Crosshair size={18} />
            <h2>Top Evacuation Targets</h2>
          </div>
          <p>Sorted by CUDA-generated risk score.</p>

          <div className="tc-target-list">
            {visibleTargets.map((target) => (
              <article className={`tc-target-row tc-severity-${target.category}`} key={target.id}>
                <div className="tc-target-rank">{target.rank ?? 'M'}</div>
                <div className="tc-target-body">
                  <div className="tc-target-top">
                    <strong>{target.source === 'MANUAL_TAG' ? 'Manual Tag' : 'Soldier ID'}: {target.soldierId ?? target.id}</strong>
                    <em>{Number.isFinite(target.risk) ? (target.risk * 100).toFixed(1) : 'UNLINKED'}</em>
                  </div>
                  <div className="tc-target-vitals">
                    {target.source === 'YOLO' ? <span>YOLO confidence: {((target.confidence ?? 0) * 100).toFixed(1)}%</span> : <span>HR: {target.hr || 'N/A'} bpm</span>}
                    {target.source === 'MANUAL_TAG' ? <span>{target.telemetryStatus ?? 'UNLINKED'} {target.bandId ? `- ${target.bandId}` : ''}</span> : target.source === 'YOLO' ? <span>{target.recommendedAction}</span> : <span>SpO2: {target.spo2}%</span>}
                  </div>
                  <div className="tc-target-bar">
                    <i style={{ width: `${Math.round((target.risk ?? 0) * 100)}%` }} />
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
  const formatTop10Overlap = (value) => {
    if (value === undefined || value === null || value === '') return 'N/A';
    const text = String(value);
    return text.includes('/') ? text : `${text}/10`;
  };
  const benchmarkSource = benchmarks ?? benchmarkRows;
  const maxSpeedup = benchmarkSource.reduce((max, row) => {
    const value = Number(row.speedup ?? 0);
    return Number.isFinite(value) && value > max ? value : max;
  }, 0);
  const speedupLabel = maxSpeedup ? `Up to ${Math.round(maxSpeedup)}x Speedup` : 'CUDA Speedup';
  const chartData = useMemo(
    () => {
      return benchmarkSource.map((row) => ({
        soldiers: row.soldiers,
        CPU: row.cpu ?? row.CPU,
        'CUDA Tiled': row.gpu ?? row.GPU,
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
        top10Overlap: formatTop10Overlap(lastBenchmark.top10Overlap),
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
      <PageExplainer tags={['Real / Implemented', 'Colab Fallback', 'Local GPU Mode']}>
        Displays verified CUDA Attention benchmark outputs generated by resqvision.cu and imported as JSON.
      </PageExplainer>

      <div className="content-grid two-columns">
        <section className="panel chart-panel">
          <div className="panel-title">
            <h3>CPU vs CUDA Runtime</h3>
            <span>Milliseconds by scenario size</span>
          </div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#dbe7f5" />
              <XAxis dataKey="soldiers" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="CPU" fill="#93c5fd" radius={[6, 6, 0, 0]} />
              <Bar dataKey="CUDA Tiled" fill="#2563eb" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </section>
        <section className="panel chart-panel">
          <div className="panel-title">
            <h3>{speedupLabel}</h3>
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
          <p className="table-note">Best verified CUDA tiled result exported from Colab benchmark sweep.</p>
          <table>
            <thead>
              <tr>
                <th>Soldiers</th>
                <th>CPU ms</th>
                <th>CUDA Tiled ms</th>
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
          <p className="metrics-subtitle">
            Status PASS means CUDA output matches the CPU reference within tolerance. Top-10 overlap compares CPU and CUDA triage priority ranking. Max and mean absolute error validate numerical accuracy.
          </p>
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

      <section className="panel demo-flow-card">
        <div>
          <h3>Demo Flow</h3>
          <p>How this dashboard is generated.</p>
        </div>
        <ol>
          <li>CUDA / Colab generates benchmark and risk data.</li>
          <li>JSON files are imported into <code>frontend/public/data</code>.</li>
          <li>Dashboard visualizes triage ranking and tactical map data.</li>
          <li>Optional image/camera input simulates UAV visual detection.</li>
        </ol>
        <p className="gpu-note">
          Local GPU execution requires an NVIDIA GPU, NVIDIA driver, CUDA Toolkit / nvcc, and MSVC Build Tools on Windows. If unavailable, use the Google Colab workflow and import <code>resqvision_cuda_outputs.zip</code>.
        </p>
      </section>
    </section>
  );
}

function SystemArchitecture() {
  const realPipelineSteps = [
    'CUDA / Colab Benchmark',
    'CSV Artifacts',
    'JSON Conversion',
    'React dataLoader',
    'Analytics',
    'Risk Ranking',
    'Tactical Map',
    'Command Decision View'
  ];
  const pipelineDetails = [
    ['Academic core', 'resqvision.cu runs attention benchmark and risk scoring.', 'core'],
    ['Real / Implemented', 'Benchmark and sweep outputs are exported for conversion.', 'real'],
    ['Real / Implemented', 'CSV files become frontend-ready JSON data.', 'real'],
    ['Real / Implemented', 'Dashboard reads frontend/public/data.', 'real'],
    ['Real / Implemented', 'Charts show timing, speedup, and correctness.', 'real'],
    ['Real / Implemented', 'Priority rows drive triage display and targets.', 'real'],
    ['Demo / Simulated', 'Targets and optional visual detections become markers.', 'demo'],
    ['Demo / Simulated', 'Recommended actions are presented for the project demo.', 'demo']
  ];
  const cudaFlow = ['resqvision.cu', 'benchmark_results.json', 'risk_ranking.json', 'attention_stats.json', 'React dashboard', 'Tactical Command'];
  const cvFlow = ['image / camera input', 'optional YOLO backend', 'detections.json / tactical_fusion.json', 'visual overlay', 'tactical map markers'];

  return (
    <section>
      <PageHeader
        eyebrow="Pipeline architecture"
        title="System Architecture"
        description="A clear view of what is implemented, what is simulated, and how data moves through the project."
      />
      <PageExplainer tags={['Real / Implemented', 'Demo / Simulated', 'Optional Backend Required']}>
        The CUDA data path is the project core. The UAV visual layer is optional and feeds the same dashboard only when demo data or a local backend is available.
      </PageExplainer>

      <section className="panel pipeline-panel architecture-pipeline">
        <div className="panel-title">
          <h3>Operational Pipeline</h3>
          <span>CUDA to dashboard</span>
        </div>
        <div className="architecture-steps">
          {realPipelineSteps.map((step, index) => (
            <article className={`architecture-step step-${pipelineDetails[index][2]}`} key={step}>
              <span>{index + 1}</span>
              <div>
                <strong>{step}</strong>
                <p>{pipelineDetails[index][1]}</p>
                <em>{pipelineDetails[index][0]}</em>
              </div>
            </article>
          ))}
        </div>
      </section>

      <div className="content-grid two-columns">
        <section className="panel data-flow-panel">
          <Network size={28} />
          <h3>Flow 1 - CUDA Data Flow</h3>
          <div className="flow-chain">
            {cudaFlow.map((step) => <span key={step}>{step}</span>)}
          </div>
          <div style={{ display: 'none', alignItems: 'center', gap: '10px', flexWrap: 'wrap', marginTop: '18px' }}>
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
        <section className="panel data-flow-panel">
          <Eye size={28} />
          <h3>Flow 2 - Computer Vision Demo Flow</h3>
          <div className="flow-chain flow-chain-optional">
            {cvFlow.map((step) => <span key={step}>{step}</span>)}
          </div>
        </section>
      </div>

      <div className="content-grid two-columns">
        <section className="panel cuda-panel">
          <Cpu size={28} />
          <h3>What is real now</h3>
          <ul>
            <li>Verified CUDA benchmark output</li>
            <li>CUDA Basic + CUDA Tiled comparison</li>
            <li>CPU/GPU correctness validation</li>
            <li>Risk ranking JSON</li>
            <li>React dashboard visualization</li>
            <li>Human-reviewed visual demo preview</li>
          </ul>
        </section>
        <section className="panel cuda-panel simulated-panel">
          <Activity size={28} />
          <h3>What is simulated / optional</h3>
          <ul>
            <li>Live UAV video feed</li>
            <li>Live YOLO backend unless scripts/yolo_server.py is running</li>
            <li>Real ResQBand telemetry stream</li>
            <li>Automated evacuation dispatch</li>
            <li>Clinical approval</li>
          </ul>
        </section>
      </div>

      <section className="panel execution-mode-panel">
        <div className="panel-title">
          <h3>Execution Modes</h3>
          <span>Choose based on hardware</span>
        </div>
        <div className="execution-mode-grid">
          <article>
            <strong>Local GPU Mode</strong>
            <p>NVIDIA GPU, driver, CUDA Toolkit / nvcc, and MSVC Build Tools on Windows.</p>
            <code>scripts/check_cuda.ps1</code>
            <code>scripts/run_cuda_local.ps1</code>
          </article>
          <article>
            <strong>Colab Fallback Mode</strong>
            <p>Use when local CUDA is unavailable. Run the notebook with GPU runtime and import the exported ZIP locally.</p>
            <code>ResQVision_Colab_Workflow.ipynb</code>
            <code>scripts/import_colab_outputs.ps1</code>
          </article>
          <article>
            <strong>Computer Vision Local Mode</strong>
            <p>Use only when the optional detection backend is running.</p>
            <code>python scripts/yolo_server.py</code>
          </article>
        </div>
      </section>
    </section>
  );
}

const CV_MOCK = [
  { id: 1, class: 'person', confidence: 0.94, bbox: [120, 80, 200, 340], center: [220, 250] },
  { id: 2, class: 'person', confidence: 0.87, bbox: [310, 95, 180, 310], center: [400, 250] },
  { id: 3, class: 'person', confidence: 0.76, bbox: [520, 110, 165, 290], center: [602, 255] }
];

const MANUAL_POINTS_STORAGE_KEY = 'resqvision_manual_drone_points';
const MAP_SIZE = 1000;

function confidenceClass(c) {
  if (c >= 0.85) return 'cv-conf-green';
  if (c >= 0.65) return 'cv-conf-orange';
  return 'cv-conf-red';
}

function normalizeManualCoordinate(value, size) {
  return Math.round((value / Math.max(size, 1)) * MAP_SIZE * 10) / 10;
}

function ComputerVision({ manualDronePoints, setManualDronePoints, refreshTacticalData }) {
  const videoRef = useRef(null);
  const [detections, setDetections] = useState(CV_MOCK);
  const [detectionSource, setDetectionSource] = useState(null);
  const [hasLiveData, setHasLiveData] = useState(false);
  const [previewVersion, setPreviewVersion] = useState(Date.now());
  const [previewAvailable, setPreviewAvailable] = useState(true);
  const [humanReviewData, setHumanReviewData] = useState(null);
  const [humanReviewAvailable, setHumanReviewAvailable] = useState(false);
  const [humanReviewWarning, setHumanReviewWarning] = useState(false);
  const [manualImageUrl, setManualImageUrl] = useState(null);
  const [manualImageName, setManualImageName] = useState(null);
  const [manualImageSize, setManualImageSize] = useState({ width: 0, height: 0 });
  const [selectedImageFile, setSelectedImageFile] = useState(null);
  const [selectedMarkerId, setSelectedMarkerId] = useState(null);
  const [isRunningYolo, setIsRunningYolo] = useState(false);
  const [isSavingMarkers, setIsSavingMarkers] = useState(false);
  const [isExportingDataset, setIsExportingDataset] = useState(false);
  const [datasetExport, setDatasetExport] = useState(null);
  const [backendError, setBackendError] = useState(null);
  const [serverStatus, setServerStatus] = useState(null);
  const [yoloNoDetections, setYoloNoDetections] = useState(false);
  const [cameraStream, setCameraStream] = useState(null);
  const [cameraError, setCameraError] = useState(null);
  const [backendOnline, setBackendOnline] = useState(false);
  const [trainingToolsOpen, setTrainingToolsOpen] = useState(false);

  // --- Pipeline state ---
  const [batchQueue, setBatchQueue] = useState([]);        // File[]
  const [batchIndex, setBatchIndex] = useState(0);         // current position in queue
  const [pipelineStep, setPipelineStep] = useState(null);  // null | 1 | 2 | 3 | 4
  const [pipelineJobId, setPipelineJobId] = useState(null);
  const [pipelineLog, setPipelineLog] = useState([]);      // string[]
  const [trainingDone, setTrainingDone] = useState(false);
  const [buildSummary, setBuildSummary] = useState(null);  // final stats
  const [advancedOpen, setAdvancedOpen] = useState(false); // UI toggle
  
  // Keep review mode states for advanced tools
  const [batchMode, setBatchMode] = useState(null);
  const [reviewYoloDetections, setReviewYoloDetections] = useState(0);
  const [savedSamplesCount, setSavedSamplesCount] = useState(0);

  // --- Debug benchmark state ---
  const [detectionMetadata, setDetectionMetadata] = useState(null);
  const [expectedSoldiers, setExpectedSoldiers] = useState("");
  const [benchmarks, setBenchmarks] = useState([]);

  useEffect(() => {
    fetch('/data/benchmarks.json?t=' + Date.now())
      .then((r) => r.json())
      .then((data) => setBenchmarks(data))
      .catch(() => setBenchmarks([]));
  }, []);

  useEffect(() => {
    const key = `resqvision_expected_${manualImageName || 'default'}`;
    const saved = localStorage.getItem(key);
    if (saved) setExpectedSoldiers(saved);
    else setExpectedSoldiers("");
  }, [manualImageName]);

  const handleExpectedSoldiersChange = (e) => {
    const val = e.target.value;
    setExpectedSoldiers(val);
    const key = `resqvision_expected_${manualImageName || 'default'}`;
    localStorage.setItem(key, val);
  };

  const handleSaveBenchmark = async () => {
    if (!expectedSoldiers || !detectionMetadata) return;
    const final_count = detectionMetadata.final_count ?? 0;
    const expected = parseInt(expectedSoldiers, 10);
    const recall = expected > 0 ? (final_count / expected) : 0;
    
    const payload = {
      image_filename: manualImageName || 'detection_preview.jpg',
      expected_soldiers: expected,
      detected_soldiers: final_count,
      estimated_recall: recall,
      model_name: detectionMetadata.model || 'unknown',
      tile_size: detectionMetadata.tile_size || 640,
      overlap_ratio: detectionMetadata.overlap || 0.35,
      min_rel_area: 0.00005 // Hardcoded for now
    };

    try {
      const res = await fetch('http://127.0.0.1:8000/api/benchmark/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (res.ok) {
        fetch('/data/benchmarks.json?t=' + Date.now())
          .then((r) => r.json())
          .then((data) => setBenchmarks(data))
          .catch(() => {});
      }
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    let liveLoaded = false;

    const loadLiveDetections = () => {
      fetch('/data/detections.json?t=' + Date.now())
        .then((r) => {
          if (!r.ok) throw new Error('detections missing');
          return r.json();
        })
        .then((data) => {
          const nextDetections = Array.isArray(data) ? data : data?.detections;
          setDetections(Array.isArray(nextDetections) ? nextDetections : CV_MOCK);
          setDetectionSource(Array.isArray(data) ? 'legacy_array' : data?.source ?? null);
          setDetectionMetadata(data?.metadata ?? null);
          setHasLiveData(true);
          liveLoaded = true;
          setPreviewVersion(Date.now());
          setPreviewAvailable(true);
        })
        .catch(() => {
          if (!liveLoaded) {
            setDetections(CV_MOCK);
            setDetectionSource(null);
            setHasLiveData(false);
          }
          // if liveLoaded is true, keep previous detections – no flicker
        });
    };

    loadLiveDetections();
    const interval = setInterval(loadLiveDetections, 1000);
    return () => clearInterval(interval);
  }, []); // empty deps – interval created once, liveLoaded via closure

  useEffect(() => {
    fetch('/data/human_review_detections.json?t=' + Date.now())
      .then((r) => {
        if (!r.ok) throw new Error('human-reviewed output missing');
        return r.json();
      })
      .then((data) => {
        setHumanReviewData(data);
        setHumanReviewAvailable(true);
        setHumanReviewWarning(false);
      })
      .catch(() => {
        setHumanReviewData(null);
        setHumanReviewAvailable(false);
        setHumanReviewWarning(true);
      });
  }, []);

  const isOfflineImage = detectionSource === 'offline_image';
  const isBrowserTransformers = detectionSource === 'browser_transformers';
  const sourceLabel = isOfflineImage ? 'Offline Image Mode' : hasLiveData ? 'Live JSON' : 'Mock data';
  const showRawYoloDebug = new URLSearchParams(window.location.search).get('debug') === '1';
  const primaryPreviewSrc = humanReviewAvailable
    ? `/data/human_review_preview.jpg?t=${previewVersion}`
    : `/data/detection_preview.jpg?t=${previewVersion}`;
  const primaryDetectionCount = humanReviewAvailable
    ? humanReviewData?.detection_count ?? humanReviewData?.detections?.length ?? 0
    : hasLiveData ? detections.length : 'Not available';
  const primarySource = humanReviewAvailable
    ? 'Human-reviewed demo'
    : isBrowserTransformers ? 'Browser Transformers.js' : detectionSource ?? 'Raw YOLO output';
  const visualSource = selectedImageFile
    ? (manualImageName?.startsWith('camera-uav-frame-') ? 'Camera frame' : 'Uploaded image')
    : humanReviewAvailable ? 'Human-reviewed demo' : hasLiveData ? 'Live JSON' : 'Demo preview';
  const outputMode = backendOnline
    ? selectedImageFile ? 'YOLO inference ready' : 'Demo preview'
    : manualDronePoints.length ? 'Manual review' : 'Demo preview';
  const currentMode = backendOnline && selectedImageFile ? 'Live YOLO Detection' : 'Demo Preview';

  useEffect(() => {
    let cancelled = false;
    const checkBackend = () => {
      fetch('http://127.0.0.1:8000/api/health')
        .then((response) => {
          if (!cancelled) setBackendOnline(response.ok);
        })
        .catch(() => {
          if (!cancelled) setBackendOnline(false);
        });
    };
    checkBackend();
    const interval = setInterval(checkBackend, 5000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    if (videoRef.current && cameraStream) {
      videoRef.current.srcObject = cameraStream;
    }
  }, [cameraStream]);

  useEffect(() => {
    return () => {
      cameraStream?.getTracks().forEach((track) => track.stop());
    };
  }, [cameraStream]);

  // Load a single File object into the image viewer (shared by single & batch).
  const loadFileIntoViewer = (file) => {
    setSelectedImageFile(file);
    setBackendError(null);
    setServerStatus(null);
    setYoloNoDetections(false);
    setManualDronePoints([]);
    setSelectedMarkerId(null);
    setManualImageUrl((currentUrl) => {
      if (currentUrl) URL.revokeObjectURL(currentUrl);
      return URL.createObjectURL(file);
    });
    setManualImageName(file.name);
    setManualImageSize({ width: 0, height: 0 });
  };

  const resetPipelineState = () => {
    setPipelineStep(null);
    setPipelineJobId(null);
    setPipelineLog([]);
    setTrainingDone(false);
    setBuildSummary(null);
    setBackendError(null);
  };

  const handleManualImageChange = (event) => {
    const files = Array.from(event.target.files ?? []);
    if (!files.length) return;

    resetPipelineState();

    if (files.length === 1) {
      // Single-image: preserve existing single-image workflow.
      setBatchQueue([files[0]]);
      setBatchIndex(0);
      loadFileIntoViewer(files[0]);
    } else {
      // Multi-image: populate the batch queue but don't auto-process.
      const sorted = [...files].sort((a, b) => a.name.localeCompare(b.name));
      setBatchQueue(sorted);
      setBatchIndex(0);
      loadFileIntoViewer(sorted[0]);
    }
  };

  const startCameraInput = async () => {
    setCameraError(null);
    setBackendError(null);

    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError('Camera input is not supported in this browser. Use image upload for the demo input.');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      setCameraStream(stream);
    } catch (error) {
      setCameraError(error.message || 'Camera permission was denied. Use image upload for the demo input.');
    }
  };

  const stopCameraInput = () => {
    cameraStream?.getTracks().forEach((track) => track.stop());
    setCameraStream(null);
  };

  const captureCameraFrame = () => {
    const video = videoRef.current;
    if (!video || !video.videoWidth || !video.videoHeight) {
      setCameraError('Camera frame is not ready yet.');
      return;
    }

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (!blob) {
        setCameraError('Could not capture camera frame.');
        return;
      }
      const file = new File([blob], `camera-uav-frame-${Date.now()}.jpg`, { type: 'image/jpeg' });
      resetPipelineState();
      setBatchQueue([file]);
      setBatchIndex(0);
      loadFileIntoViewer(file);
      setServerStatus('Camera frame captured for UAV-style simulation input.');
    }, 'image/jpeg', 0.92);
  };

  const handleManualImageLoad = (event) => {
    setManualImageSize({
      width: event.currentTarget.naturalWidth,
      height: event.currentTarget.naturalHeight
    });
  };

  const handleManualImageClick = (event) => {
    const image = event.currentTarget;
    const rect = image.getBoundingClientRect();
    const imageWidth = manualImageSize.width || image.naturalWidth;
    const imageHeight = manualImageSize.height || image.naturalHeight;
    if (!imageWidth || !imageHeight || !rect.width || !rect.height) return;

    const x = Math.round(((event.clientX - rect.left) / rect.width) * imageWidth);
    const y = Math.round(((event.clientY - rect.top) / rect.height) * imageHeight);
    const xMap = normalizeManualCoordinate(x, imageWidth);
    const yMap = normalizeManualCoordinate(y, imageHeight);
    const xNorm = Math.round((x / Math.max(imageWidth, 1)) * 10000) / 10000;
    const yNorm = Math.round((y / Math.max(imageHeight, 1)) * 10000) / 10000;

    if (selectedMarkerId) {
      setManualDronePoints((current) => current.map((point) => (
        point.id === selectedMarkerId
          ? {
              ...point,
              x_image: x,
              y_image: y,
              x_norm: xNorm,
              y_norm: yNorm,
              image_center: [x, y],
              x_map: xMap,
              y_map: yMap,
              map_position: [xMap, yMap],
            }
          : point
      )));
      setSelectedMarkerId(null);
      return;
    }

    const nextIndex = manualDronePoints.length + 1;

    setManualDronePoints((current) => [
      ...current,
      {
        id: `manual_${nextIndex}`,
        source: 'MANUAL_TAG',
        soldier_id: '',
        band_id: '',
        class: 'visual_point',
        confidence: 1.0,
        x_image: x,
        y_image: y,
        x_norm: xNorm,
        y_norm: yNorm,
        image_center: [x, y],
        x_map: xMap,
        y_map: yMap,
        map_position: [xMap, yMap],
        localization_mode: 'manual_visual_relative',
        localization_label: 'Manual Drone Visual Fix',
        label: 'Soldier',
        status: '',
        priority: null,
        risk_score: null
      }
    ]);
  };

  const clearManualPoints = () => {
    setManualDronePoints([]);
    setSelectedMarkerId(null);
  };

  const updateManualPoint = (id, patch) => {
    setManualDronePoints((current) => current.map((point) => (
      point.id === id ? { ...point, ...patch } : point
    )));
  };

  const deleteManualPoint = (id) => {
    setManualDronePoints((current) => current.filter((point) => point.id !== id));
    if (selectedMarkerId === id) setSelectedMarkerId(null);
  };

  const refreshDetectionsOnce = () => {
    return fetch('/data/detections.json?t=' + Date.now())
      .then((r) => {
        if (!r.ok) throw new Error('detections missing');
        return r.json();
      })
      .then((data) => {
        const nextDetections = Array.isArray(data) ? data : data?.detections;
        setDetections(Array.isArray(nextDetections) ? nextDetections : CV_MOCK);
        setDetectionSource(Array.isArray(data) ? 'legacy_array' : data?.source ?? null);
        setDetectionMetadata(data?.metadata ?? null);
        setHasLiveData(true);
        setPreviewVersion(Date.now());
        setPreviewAvailable(true);
      });
  };

  // Upload a single file to the YOLO endpoint and refresh detection state.
  const uploadFileToYolo = async (file) => {
    const formData = new FormData();
    formData.append('image', file);
    const response = await fetch('http://127.0.0.1:8000/api/yolo/upload', {
      method: 'POST',
      body: formData,
    });
    const result = await response.json();
    if (!response.ok || !result.ok) {
      throw new Error(result.message || 'YOLO upload failed');
    }
    return result;
  };

  const runYoloOnUploadedImage = async () => {
    if (!selectedImageFile) {
      setBackendError('Upload a drone image before running YOLO.');
      return;
    }

    setIsRunningYolo(true);
    setBackendError(null);
    setServerStatus(null);
    setYoloNoDetections(false);

    try {
      const result = await uploadFileToYolo(selectedImageFile);
      setYoloNoDetections(result.detections === 0);
      setServerStatus(`YOLO complete: ${result.detections} detection${result.detections === 1 ? '' : 's'} fused.`);
      await refreshDetectionsOnce();
      refreshTacticalData?.();
    } catch (error) {
      setBackendError(error.message || 'Local YOLO backend is not running.');
    } finally {
      setIsRunningYolo(false);
    }
  };

  // Save current markers then advance to the next image in the batch.
  const saveMarkersForBatchImage = async () => {
    const markers = manualDronePoints.map((point) => ({
      source: 'MANUAL_TAG',
      soldier_id: point.soldier_id ?? '',
      band_id: point.band_id ?? '',
      x_image: point.x_image ?? point.image_center?.[0] ?? 0,
      y_image: point.y_image ?? point.image_center?.[1] ?? 0,
      x_norm: point.x_norm ?? 0,
      y_norm: point.y_norm ?? 0,
      x_map: point.x_map ?? point.map_position?.[0] ?? 0,
      y_map: point.y_map ?? point.map_position?.[1] ?? 0,
      label: point.label ?? 'Soldier',
      status: point.status ?? '',
    }));

    if (!markers.length) return { ok: true, dataset_sample: { created: false } };

    const response = await fetch('http://127.0.0.1:8000/api/markers/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(markers),
    });
    return response.json();
  };

  // ── One-Button Pipeline: Build & Train
  const buildAndTrain = async () => {
    if (!batchQueue.length) {
      setBackendError('Select images first.');
      return;
    }
    
    // Reset state
    setPipelineStep(1);
    setPipelineJobId(null);
    setPipelineLog([]);
    setTrainingDone(false);
    setBuildSummary(null);
    setBackendError(null);

    let totalDetections = 0;
    let zeroDetectionImages = 0;
    let savedCount = 0;

    // STEP 1: Process images
    for (let i = 0; i < batchQueue.length; i++) {
      const file = batchQueue[i];
      setBatchIndex(i);
      setManualImageUrl((url) => { if (url) URL.revokeObjectURL(url); return URL.createObjectURL(file); });
      setManualImageName(file.name);
      
      try {
        const formData = new FormData();
        formData.append('image', file);
        const response = await fetch('http://127.0.0.1:8000/api/yolo/upload_and_save', {
          method: 'POST',
          body: formData,
        });
        const result = await response.json();
        if (!response.ok || !result.ok) throw new Error(result.message || 'Upload failed');

        const det = result.detections ?? 0;
        totalDetections += det;
        savedCount += 1;
        if (det === 0) zeroDetectionImages += 1;
      } catch (err) {
        setBackendError(`Image ${i + 1} failed: ${err.message}`);
      }
    }

    // STEP 2: Export & Launch Training
    setPipelineStep(2);
    let jobId = null;
    let exportResult = null;
    try {
      const resp = await fetch('http://127.0.0.1:8000/api/pipeline/build-and-train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ epochs: 3, batch: 4 })
      });
      const data = await resp.json();
      if (!resp.ok || !data.ok) throw new Error(data.message || 'Failed to start pipeline');
      
      exportResult = data.dataset;

      if (data.status === 'local_training_skipped_colab_required') {
        setPipelineStep(4);
        setTrainingDone(true);
        setBuildSummary({
          status: 'local_training_skipped_colab_required',
          images: savedCount,
          labels: totalDetections,
          zeroDetections: zeroDetectionImages,
          zipPath: data.dataset_zip_path ?? 'dataset.zip',
          exportOk: !!exportResult?.ok,
          totalInQueue: batchQueue.length
        });
        refreshTacticalData?.();
        return;
      }
      
      jobId = data.job_id;
      setPipelineJobId(jobId);
    } catch (err) {
      setBackendError(`Pipeline start failed: ${err.message}`);
      setPipelineStep(null);
      return;
    }

    // STEP 3: Poll Training Status
    setPipelineStep(3);
    
    const pollInterval = setInterval(async () => {
      try {
        const statusResp = await fetch(`http://127.0.0.1:8000/api/pipeline/status/${jobId}`);
        const statusData = await statusResp.json();
        
        if (statusData.ok) {
          if (statusData.log_tail && statusData.log_tail.length > 0) {
            setPipelineLog(prev => [...prev, ...statusData.log_tail]);
          }
          
          if (!statusData.running) {
            clearInterval(pollInterval);
            // STEP 4: Done
            setPipelineStep(4);
            setTrainingDone(true);
            setBuildSummary({
              status: statusData.returncode === 0 ? 'local_training_completed' : 'local_training_failed',
              images: savedCount,
              labels: totalDetections,
              zeroDetections: zeroDetectionImages,
              zipPath: exportResult?.zip_path ?? 'dataset.zip',
              exportOk: !!exportResult?.ok,
              totalInQueue: batchQueue.length,
              returncode: statusData.returncode
            });
            refreshTacticalData?.();
          }
        }
      } catch (err) {
        console.error("Polling error", err);
      }
    }, 3000);
  };

  // ── Review batch: open image 1, run YOLO, wait for user, Save & Next moves to image 2, etc.
  const startReview = async () => {
    if (!batchQueue.length) {
      setBackendError('No batch queue loaded. Select multiple images first.');
      return;
    }
    setBatchMode('review');
    setSavedSamplesCount(0);
    setBatchIndex(0);
    setReviewYoloDetections(0);
    setBackendError(null);
    const file = batchQueue[0];
    loadFileIntoViewer(file);
    setServerStatus(`Reviewing image 1 / ${batchQueue.length}: ${file.name} — running YOLO…`);
    // Run YOLO on the first image so detections are ready.
    try {
      const result = await uploadFileToYolo(file);
      await refreshDetectionsOnce();
      refreshTacticalData?.();
      setReviewYoloDetections(result.detections);
      setServerStatus(
        `Reviewing image 1 / ${batchQueue.length} — YOLO: ${result.detections} detection(s). ` +
        `Add missing soldiers, then click "Save & Next".`
      );
    } catch (err) {
      setBackendError(`YOLO failed for image 1: ${err.message}`);
    }
  };

  // Save current image (YOLO boxes + manual tags) and advance to next review image.
  const saveAndNext = async () => {
    if (batchMode !== 'review') return;
    setIsSavingMarkers(true);
    setBackendError(null);

    try {
      // Serialize current manual tags to pass alongside the image.
      const manualPayload = manualDronePoints.map((point) => ({
        source: 'MANUAL_TAG',
        soldier_id: point.soldier_id ?? '',
        band_id: point.band_id ?? '',
        x_image: point.x_image ?? point.image_center?.[0] ?? 0,
        y_image: point.y_image ?? point.image_center?.[1] ?? 0,
        x_norm: point.x_norm ?? 0,
        y_norm: point.y_norm ?? 0,
        x_map: point.x_map ?? point.map_position?.[0] ?? 0,
        y_map: point.y_map ?? point.map_position?.[1] ?? 0,
        label: point.label ?? 'Soldier',
        status: point.status ?? '',
      }));

      // Call the atomic endpoint — it re-uploads the current file so it uses
      // the in-memory image for label writing (not the shared drone_frame.png).
      const formData = new FormData();
      formData.append('image', batchQueue[batchIndex]);
      if (manualPayload.length) {
        formData.append('manual_markers', JSON.stringify(manualPayload));
      }
      const response = await fetch('http://127.0.0.1:8000/api/yolo/upload_and_save', {
        method: 'POST',
        body: formData,
      });
      const result = await response.json();
      if (!response.ok || !result.ok) throw new Error(result.message || 'Save failed');

      const sample = result.dataset_sample;
      const newSaved = savedSamplesCount + (sample?.created ? 1 : 0);
      setSavedSamplesCount(newSaved);

      const savedLabel = sample?.created ? `→ ${sample.frame}` : '(no labels, skipped)';
      setServerStatus(`Saved image ${batchIndex + 1} / ${batchQueue.length} ${savedLabel}.`);
      refreshTacticalData?.();

      const nextIdx = batchIndex + 1;
      if (nextIdx < batchQueue.length) {
        // Advance to next image and run YOLO on it.
        setBatchIndex(nextIdx);
        const nextFile = batchQueue[nextIdx];
        loadFileIntoViewer(nextFile);
        setReviewYoloDetections(0);
        setServerStatus(`Reviewing image ${nextIdx + 1} / ${batchQueue.length}: ${nextFile.name} — running YOLO…`);
        try {
          const nextResult = await uploadFileToYolo(nextFile);
          await refreshDetectionsOnce();
          setReviewYoloDetections(nextResult.detections);
          setServerStatus(
            `Reviewing image ${nextIdx + 1} / ${batchQueue.length} — ` +
            `YOLO: ${nextResult.detections} detection(s). Add missing soldiers, then click "Save & Next".`
          );
        } catch (err) {
          setBackendError(`YOLO failed for image ${nextIdx + 1}: ${err.message}`);
        }
      } else {
        // All images reviewed.
        setBatchMode(null);
        setServerStatus(
          `Review complete. ${newSaved} / ${batchQueue.length} samples saved. ` +
          `Click "Export Training Dataset" to package them.`
        );
      }
    } catch (err) {
      setBackendError(err.message || 'Save failed.');
    } finally {
      setIsSavingMarkers(false);
    }
  };

  const saveManualMarkers = async () => {
    setIsSavingMarkers(true);
    setBackendError(null);
    setServerStatus(null);

    try {
      if (selectedImageFile) {
        try {
          const formData = new FormData();
          formData.append('image', selectedImageFile);
          await fetch('http://127.0.0.1:8000/api/yolo/upload', {
            method: 'POST',
            body: formData,
          });
        } catch {
          // Marker saving should still work even if the optional upload/YOLO pass fails.
        }
      }

      const markers = manualDronePoints.map((point) => ({
        source: 'MANUAL_TAG',
        soldier_id: point.soldier_id ?? '',
        band_id: point.band_id ?? '',
        x_image: point.x_image ?? point.image_center?.[0] ?? 0,
        y_image: point.y_image ?? point.image_center?.[1] ?? 0,
        x_norm: point.x_norm ?? 0,
        y_norm: point.y_norm ?? 0,
        x_map: point.x_map ?? point.map_position?.[0] ?? 0,
        y_map: point.y_map ?? point.map_position?.[1] ?? 0,
        label: point.label ?? 'Soldier',
        status: point.status ?? '',
      }));
      const response = await fetch('http://127.0.0.1:8000/api/markers/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(markers),
      });
      const result = await response.json();
      if (!response.ok || !result.ok) {
        throw new Error(result.message || 'Marker save failed');
      }
      const sample = result.dataset_sample;
      if (sample?.created) {
        setServerStatus(`Training sample added: ${sample.frame} with ${sample.labels} label${sample.labels === 1 ? '' : 's'}.`);
      } else {
        setServerStatus('Markers saved, but no training sample was added. Upload an image and run YOLO or save again.');
      }
      refreshTacticalData?.();
    } catch (error) {
      setBackendError(error.message || 'Local YOLO backend is not running.');
    } finally {
      setIsSavingMarkers(false);
    }
  };

  const exportTrainingDataset = async () => {
    setIsExportingDataset(true);
    setBackendError(null);
    setDatasetExport(null);

    try {
      const response = await fetch('http://127.0.0.1:8000/api/dataset/export', {
        method: 'POST',
      });
      const result = await response.json();
      if (!response.ok || !result.ok) {
        throw new Error(result.message || 'Dataset export failed');
      }
      setDatasetExport(result);
      setServerStatus(`Dataset exported successfully. Images: ${result.images}. Labels: ${result.labels}.`);
    } catch (error) {
      setBackendError(error.message || 'Local YOLO backend is not running.');
    } finally {
      setIsExportingDataset(false);
    }
  };

  return (
    <section className="cv-page">
      <PageHeader
        eyebrow="Computer vision"
        title="UAV Visual Simulation"
        description="Simulate a drone-style visual input using either image upload, browser camera, or human-reviewed demo preview."
      />
      <PageExplainer tags={['Demo / Simulated', 'Optional Backend Required']}>
        Demo Preview uses a human-reviewed image. Live YOLO requires <code>python scripts/yolo_server.py</code>. CUDA benchmark data is separate and already loaded from JSON.
      </PageExplainer>

      <section className={`panel cv-mode-explainer ${backendOnline ? 'is-online' : 'is-offline'}`}>
        <h2>Current mode: {currentMode}</h2>
        {backendOnline ? (
          <p>YOLO backend is online. Upload an image or capture a camera frame, then run detection to generate live YOLO output.</p>
        ) : (
          <p>YOLO backend is offline, so this page shows a stable human-reviewed preview. This is not live YOLO output. For the full automated demo, run <code>powershell -ExecutionPolicy Bypass -File .\scripts\start_resqvision_demo.ps1 -StartYolo</code>, refresh the page, then upload an image or capture a camera frame.</p>
        )}
      </section>

      <section className={`cv-status-banner ${backendOnline ? 'is-online' : 'is-offline'}`}>
        <div>
          <span>Current mode</span>
          <strong>{backendOnline && selectedImageFile ? 'Live YOLO Detection' : 'Demo Preview'}</strong>
        </div>
        <div>
          <span>YOLO Backend</span>
          <strong>{backendOnline ? 'Online' : 'Offline'}</strong>
        </div>
        <div>
          <span>Detection</span>
          <strong>{backendOnline && selectedImageFile ? 'YOLO inference ready' : 'Human-reviewed fallback'}</strong>
        </div>
        <div>
          <span>Output</span>
          <strong>{backendOnline && selectedImageFile ? outputMode : 'Visual demo overlay'}</strong>
        </div>
      </section>

      <section className="panel cv-workflow-card">
        <div>
          <strong>Demo mode</strong>
          <span>Human-reviewed image</span>
          <i>visual overlay</i>
          <span>dashboard preview</span>
        </div>
        <div>
          <strong>Live detection mode</strong>
          <span>Upload/camera</span>
          <i>YOLO backend</i>
          <span>detections.json / tactical_fusion.json</span>
          <i>dashboard overlay</i>
        </div>
      </section>

      <section className="panel cv-steps-card">
        <article>
          <i>1</i>
          <strong>Select Visual Input</strong>
          <p>Choose demo preview, upload image, or browser camera.</p>
          <em>Current: {visualSource}</em>
        </article>
        <article>
          <i>2</i>
          <strong>Run Detection</strong>
          <p>{backendOnline ? 'Run YOLO inference on uploaded or captured input.' : 'Backend is offline, so use the human-reviewed demo preview.'}</p>
          <em>{backendOnline ? 'Action: Run Detection' : 'Fallback active'}</em>
        </article>
        <article>
          <i>3</i>
          <strong>Review and Send</strong>
          <p>Use detections for visual overlay and Tactical Command map markers.</p>
          <em>Review before dashboard use</em>
        </article>
      </section>

      <BrowserCvDetector
        backendOnline={backendOnline}
        onSaved={async () => {
          setHumanReviewAvailable(false);
          setHumanReviewWarning(false);
          await refreshDetectionsOnce();
          refreshTacticalData?.();
          setServerStatus('Browser Transformers.js detections saved and tactical fusion refreshed.');
        }}
      />

      <section className="panel cv-input-card">
        <div className="cv-section-heading">
          <h2>Visual Input Source</h2>
          <p>Select the source for the UAV-style visual layer before reviewing the current result.</p>
        </div>
        <div className={`cv-input-option cv-demo-option ${!selectedImageFile ? 'is-active' : ''}`}>
          <Eye size={20} />
          <div>
            <h3>Human-reviewed demo</h3>
            <p>Always available. Uses reviewed annotations. Recommended for stable demo.</p>
          </div>
          <button type="button" className="cv-mode-button" onClick={() => {
            setSelectedImageFile(null);
            setManualImageName(null);
            setManualImageUrl((currentUrl) => {
              if (currentUrl) URL.revokeObjectURL(currentUrl);
              return null;
            });
            setServerStatus('Human-reviewed demo preview is active.');
          }}>
            {!selectedImageFile ? 'Currently Active' : 'Use Demo Preview'}
          </button>
        </div>
        <div className="cv-input-option">
          <Upload size={20} />
          <div>
            <h3>Upload UAV-style Image</h3>
            <p>Requires YOLO backend. Upload image and run YOLO inference through the local Python backend.</p>
          </div>
          <label className="drone-file-control compact">
            <span>Select Image</span>
            <input
              type="file"
              accept="image/*"
              onChange={handleManualImageChange}
            />
          </label>
          <button
            type="button"
            className="cv-run-detection"
            onClick={runYoloOnUploadedImage}
            disabled={!backendOnline || !selectedImageFile || isRunningYolo}
          >
            {isRunningYolo ? 'Running Detection...' : 'Run Detection'}
          </button>
          {!backendOnline ? <small>Start scripts/yolo_server.py to enable detection.</small> : null}
        </div>
        <div className="cv-input-option">
          <Camera size={20} />
          <div>
            <h3>Browser Camera Simulation</h3>
            <p>Camera preview/capture is browser-side. YOLO inference requires backend.</p>
          </div>
          <div className="camera-controls">
            <button type="button" onClick={cameraStream ? stopCameraInput : startCameraInput}>
              {cameraStream ? 'Stop Camera' : 'Start Camera'}
            </button>
            <button type="button" onClick={captureCameraFrame} disabled={!cameraStream}>
              Capture Frame
            </button>
          </div>
          {!backendOnline ? <small>Capture can simulate input, but detection requires YOLO backend.</small> : null}
        </div>
        {cameraStream ? (
          <video ref={videoRef} className="cv-camera-preview" autoPlay muted playsInline />
        ) : null}
        {cameraError ? <div className="cv-yolo-empty">{cameraError}</div> : null}
      </section>

      <section className="panel cv-human-review-card">
        <div className="cv-human-review-header">
          <div>
            <h2>Current Visual Result</h2>
            <p>{backendOnline && selectedImageFile ? 'Uploaded or captured visual input ready for YOLO inference.' : 'This preview is human-reviewed and is shown because live YOLO inference is currently unavailable. This is not live YOLO output. It is a human-reviewed demo preview shown while the YOLO backend is offline.'}</p>
          </div>
          <dl className="cv-human-review-summary readable-summary">
            <div><dt>Source</dt><dd>{primarySource}</dd></div>
            <div><dt>Detections</dt><dd>{primaryDetectionCount}</dd></div>
            <div><dt>Review status</dt><dd>{humanReviewAvailable ? 'Human reviewed' : 'Fallback preview'}</dd></div>
            <div><dt>Used for</dt><dd>Demo visualization</dd></div>
          </dl>
        </div>

        {humanReviewWarning ? (
          <div className="cv-human-review-warning">
            Human-reviewed demo output not found. Showing raw detection output.
          </div>
        ) : null}

        {previewAvailable || humanReviewAvailable ? (
          <img
            src={primaryPreviewSrc}
            alt={humanReviewAvailable ? 'Human-reviewed tactical annotation preview' : 'Raw YOLO detection preview'}
            className="cv-human-review-image"
            onError={() => {
              if (humanReviewAvailable) {
                setHumanReviewAvailable(false);
                setHumanReviewWarning(true);
                setPreviewVersion(Date.now());
              } else {
                setPreviewAvailable(false);
              }
            }}
          />
        ) : (
          <div className="cv-image-placeholder">
            <div>
              <p>No computer vision preview found</p>
              <small>Generate <code>human_review_preview.jpg</code> or run YOLO to create <code>detection_preview.jpg</code>.</small>
            </div>
          </div>
        )}
      </section>

      <section className={`cv-backend-guidance ${backendOnline ? 'is-online' : ''}`}>
        <div>
          <strong>{backendOnline ? 'YOLO Backend Online' : 'YOLO Backend Not Running - Demo Fallback Active'}</strong>
          {backendOnline ? (
            <p>YOLO inference can run against the local server. Upload or capture a frame, then run detection.</p>
          ) : (
            <>
              <p>YOLO backend is offline. For the full automated demo, run:</p>
              <p><code>powershell -ExecutionPolicy Bypass -File .\scripts\start_resqvision_demo.ps1 -StartYolo</code></p>
              <p>Manual local YOLO option:</p>
              <ol>
                <li>Make sure your Python environment is active.</li>
                <li>Run: <code>python scripts/yolo_server.py</code></li>
                <li>Refresh this page.</li>
                <li>Upload an image or capture a frame.</li>
              </ol>
              <p>The page will continue using the human-reviewed fallback preview.</p>
            </>
          )}
        </div>
      </section>

      <div className="cv-pipeline-layout">
        <details className="panel pipeline-panel cv-training-details" open={trainingToolsOpen} onToggle={(event) => setTrainingToolsOpen(event.currentTarget.open)}>
          <summary>
            <div>
              <h3>Optional YOLO Training Pipeline</h3>
              <span>Not required for CUDA benchmark demo</span>
            </div>
            <b>{trainingToolsOpen ? 'Hide Optional Training Tools' : 'Show Optional Training Tools'}</b>
          </summary>
          <p className="manual-demo-note">
            Training is only needed to update the visual model. It is not required for CUDA benchmark or normal demo.
          </p>

          <div className="pipeline-controls">
            <label className="drone-file-control">
              <span>Select drone frames</span>
              <input
                id="drone-file-input"
                type="file"
                accept="image/*"
                multiple
                onChange={handleManualImageChange}
              />
            </label>

            {batchQueue.length > 0 && !pipelineStep && (
              <button
                type="button"
                className="batch-build-btn"
                onClick={buildAndTrain}
              >
                ⚡ Build Dataset & Train
              </button>
            )}
          </div>

          {pipelineStep && (
            <div className="pipeline-card">
              <div className={`pipeline-step ${pipelineStep === 1 ? 'active' : pipelineStep > 1 ? 'done' : ''}`}>
                <div className="step-num">{pipelineStep > 1 ? '✓' : '1'}</div>
                <div className="step-label">Processing images {pipelineStep === 1 ? `(${batchIndex + 1}/${batchQueue.length})` : ''}</div>
              </div>
              <div className={`pipeline-step ${pipelineStep === 2 ? 'active' : pipelineStep > 2 ? 'done' : ''}`}>
                <div className="step-num">{pipelineStep > 2 ? '✓' : '2'}</div>
                <div className="step-label">Exporting dataset</div>
              </div>
              <div className={`pipeline-step ${pipelineStep === 3 ? 'active' : pipelineStep > 3 ? 'done' : ''}`}>
                <div className="step-num">{pipelineStep > 3 ? '✓' : '3'}</div>
                <div className="step-label">Training model</div>
              </div>
              <div className={`pipeline-step ${pipelineStep === 4 ? 'done' : ''}`}>
                <div className="step-num">{pipelineStep === 4 ? '✓' : '4'}</div>
                <div className="step-label">Done</div>
              </div>

              {pipelineStep === 3 && pipelineLog.length > 0 && (
                <div className="pipeline-log">
                  {pipelineLog.slice(-15).map((logLine, idx) => (
                    <div key={idx}>{logLine}</div>
                  ))}
                </div>
              )}

              {buildSummary && (
                <div className="build-summary-card">
                  {buildSummary.status === 'local_training_skipped_colab_required' ? (
                    <>
                      <div className="build-summary-title">Dataset built successfully</div>
                      <dl className="build-summary-stats">
                        <div><dt>Local GPU</dt><dd>not available</dd></div>
                        <div><dt>Local training</dt><dd>skipped</dd></div>
                        <div><dt>Training mode</dt><dd>Colab required</dd></div>
                        <div><dt>Dataset Path</dt><dd>{buildSummary.zipPath}</dd></div>
                      </dl>
                      <div className="cv-note" style={{ marginTop: '12px', display: 'block' }}>
                        <strong>Next steps:</strong>
                        <ol style={{ margin: '8px 0 0 20px', padding: 0 }}>
                          <li>Open ResQVision_Colab_Workflow.ipynb</li>
                          <li>Upload dataset.zip</li>
                          <li>Train on Colab GPU</li>
                          <li>Download best.pt</li>
                          <li>Place it at models/drone_tactical_best.pt</li>
                          <li>Restart yolo_server.py</li>
                        </ol>
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="build-summary-title">
                        {buildSummary.status === 'local_training_completed' ? '✅ Training Complete' : '⚠️ Training Finished with Errors'}
                      </div>
                      <dl className="build-summary-stats">
                        <div>
                          <dt>Images processed</dt>
                          <dd>{buildSummary.images} / {buildSummary.totalInQueue}</dd>
                        </div>
                        <div>
                          <dt>Total detections saved</dt>
                          <dd>{buildSummary.labels}</dd>
                        </div>
                        <div className={buildSummary.zeroDetections > 0 ? 'build-stat-warn' : ''}>
                          <dt>Needs review (0 detections)</dt>
                          <dd>{buildSummary.zeroDetections}</dd>
                        </div>
                        <div>
                          <dt>Model Path</dt>
                          <dd>models/drone_tactical_best.pt</dd>
                        </div>
                      </dl>
                    </>
                  )}
                </div>
              )}
            </div>
          )}

          {backendError && (
            <div className="cv-backend-error">
              <strong>Error</strong>
              <small>{backendError}</small>
            </div>
          )}
        </details>
      </div>

      {showRawYoloDebug ? (
      <details className="advanced-toggle" open={advancedOpen} onToggle={e => setAdvancedOpen(e.target.open)}>
        <summary>Raw YOLO Debug Output</summary>
        <div className="advanced-content">
      <div className="cv-layout">
        <section className="panel cv-detections-panel">
          <div className="panel-title">
            <h3>Raw YOLO Debug Output</h3>
            <span>{detections.length} object{detections.length !== 1 ? 's' : ''}</span>
          </div>
          
          {detectionMetadata && (
            <div className="cv-benchmark-panel" style={{ padding: '16px', background: 'var(--bg-card)', borderBottom: '1px solid var(--border)', marginBottom: '8px' }}>
              <h4 style={{ marginTop: 0, marginBottom: '12px' }}>Detection Benchmark</h4>
              <dl style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', margin: 0, fontSize: '0.85rem' }}>
                <div><dt>Image</dt><dd>{manualImageName || 'detection_preview.jpg'}</dd></div>
                <div><dt>Raw Detections</dt><dd>{detectionMetadata.raw_detections_count ?? '-'}</dd></div>
                <div><dt>Post-Filter</dt><dd>{detectionMetadata.post_geometry_filter_count ?? '-'}</dd></div>
                <div><dt>Post-Merge</dt><dd>{detectionMetadata.post_merge_count ?? '-'}</dd></div>
                <div><dt>Post-NMS (Final)</dt><dd>{detectionMetadata.final_count ?? '-'}</dd></div>
              </dl>
              <div style={{ marginTop: '12px', display: 'flex', gap: '12px', alignItems: 'center' }}>
                <label style={{ fontSize: '0.85rem' }}>Expected soldiers:</label>
                <input 
                  type="number" 
                  style={{ width: '60px', padding: '4px' }} 
                  value={expectedSoldiers} 
                  onChange={handleExpectedSoldiersChange}
                />
                {expectedSoldiers > 0 && detectionMetadata.final_count !== undefined && (
                  <span style={{ fontWeight: 'bold', fontSize: '0.9rem', color: 'var(--text-highlight)' }}>
                    Recall: {((detectionMetadata.final_count / expectedSoldiers) * 100).toFixed(2)}%
                  </span>
                )}
                <button 
                  className="btn btn-secondary" 
                  style={{ padding: '4px 8px', fontSize: '0.8rem' }}
                  onClick={handleSaveBenchmark}
                  disabled={!expectedSoldiers || expectedSoldiers <= 0}
                >
                  Save Benchmark
                </button>
              </div>
            </div>
          )}
          
          {benchmarks.length > 0 && (
            <div className="cv-benchmark-summary" style={{ padding: '16px', background: 'var(--bg-card)', borderBottom: '1px solid var(--border)', marginBottom: '8px' }}>
              <h4 style={{ marginTop: 0, marginBottom: '12px' }}>Benchmark Summary</h4>
              <dl style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', margin: 0, fontSize: '0.85rem' }}>
                <div><dt>Images Tested</dt><dd>{benchmarks.length}</dd></div>
                <div><dt>Zero Detection</dt><dd>{benchmarks.filter(b => b.detected_soldiers === 0).length}</dd></div>
                <div><dt>Average Recall</dt><dd>{(benchmarks.reduce((a, b) => a + b.estimated_recall, 0) / benchmarks.length * 100).toFixed(2)}%</dd></div>
                <div>
                  <dt>Median Recall</dt>
                  <dd>
                    {(() => {
                      const sorted = [...benchmarks].map(b => b.estimated_recall).sort((a,b) => a-b);
                      const med = sorted.length % 2 === 0 
                        ? (sorted[sorted.length/2 - 1] + sorted[sorted.length/2]) / 2 
                        : sorted[Math.floor(sorted.length/2)];
                      return (med * 100).toFixed(2) + '%';
                    })()}
                  </dd>
                </div>
              </dl>
            </div>
          )}

          <div className="cv-detection-list">
            {detections.map((det) => (
              <article key={det.id} className="cv-detection-row">
                <div className="cv-det-id">#{det.id}</div>
                <div className="cv-det-body">
                  <div className="cv-det-top">
                    <strong>{det.class}</strong>
                    <span className={`cv-confidence ${confidenceClass(det.confidence)}`}>
                      {(det.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="cv-det-bbox">
                    bbox&nbsp;&nbsp;[
                    {det.bbox.map((v, i) => (
                      <span key={i}>{i > 0 ? ', ' : ''}{v}</span>
                    ))}
                    ]&nbsp;&nbsp;<small>x, y, w, h</small>
                  </div>
                  {Array.isArray(det.center) ? (
                    <div className="cv-det-bbox">
                      center&nbsp;[
                      {det.center.map((v, i) => (
                        <span key={i}>{i > 0 ? ', ' : ''}{v}</span>
                      ))}
                      ]
                    </div>
                  ) : null}
                  <div className="cv-conf-bar">
                    <div
                      className={`cv-conf-fill ${confidenceClass(det.confidence)}`}
                      style={{ width: `${(det.confidence * 100).toFixed(1)}%` }}
                    />
                  </div>
                </div>
              </article>
            ))}
          </div>
        </section>
      </div>

      <section className="panel drone-marking-panel">
        <div className="panel-title">
          <h3>Drone Image Marking</h3>
          <span>{manualDronePoints.length} manual tag{manualDronePoints.length !== 1 ? 's' : ''}</span>
        </div>
        <p className="manual-demo-note">
          Upload one or more drone frames. Run YOLO automatically, or review each image and add missing tags manually.
        </p>

        <div className="drone-marking-grid">
          <div className="drone-upload-column">
            <label className="drone-file-control">
              <span>Select drone frames</span>
              <input
                id="drone-file-input"
                type="file"
                accept="image/*"
                multiple
                onChange={handleManualImageChange}
              />
            </label>

            {/* Single-image manual actions */}
              <div className="drone-action-row">
                <button
                  id="btn-run-yolo"
                  type="button"
                  onClick={runYoloOnUploadedImage}
                  disabled={!selectedImageFile || isRunningYolo}
                >
                  {isRunningYolo ? 'Running YOLO...' : 'Run YOLO'}
                </button>
                <button
                  id="btn-save-tags"
                  type="button"
                  onClick={saveManualMarkers}
                  disabled={isSavingMarkers}
                >
                  {isSavingMarkers ? 'Saving...' : 'Save Tactical Tags'}
                </button>
                <button
                  id="btn-export-dataset"
                  type="button"
                  onClick={exportTrainingDataset}
                  disabled={isExportingDataset}
                >
                  {isExportingDataset ? 'Exporting...' : 'Export Training Dataset'}
                </button>
              </div>
            {backendError ? (
              <div className="cv-backend-error">
                <strong>Local YOLO backend is not running.</strong>
                <span>Start it with:</span>
                <code>venv\Scripts\python.exe scripts\yolo_server.py</code>
                <small>{backendError}</small>
              </div>
            ) : null}
            {serverStatus ? <div className="cv-server-status">{serverStatus}</div> : null}

            {/* Old Build Summary removed */}
            {yoloNoDetections ? (
              <div className="cv-yolo-empty">
                YOLO did not detect soldiers automatically in this overhead frame. Use Manual Tactical Tagging to mark soldiers and link them to ResQBand IDs.
              </div>
            ) : null}

            <div className="drone-image-stage">
              {manualImageUrl ? (
                <>
                  <img
                    src={manualImageUrl}
                    alt={manualImageName ?? 'Uploaded drone frame'}
                    className="drone-marking-image"
                    onClick={handleManualImageClick}
                    onLoad={handleManualImageLoad}
                  />
                  {manualImageSize.width > 0 ? manualDronePoints.map((point) => (
                    <span
                      key={point.id}
                      className="drone-click-marker"
                      style={{
                        left: `${(point.image_center[0] / manualImageSize.width) * 100}%`,
                        top: `${(point.image_center[1] / manualImageSize.height) * 100}%`
                      }}
                    >
                      {point.id.replace('manual_', '')}
                    </span>
                  )) : null}
                </>
              ) : (
                <div className="drone-image-placeholder">
                  <Eye size={42} strokeWidth={1.2} />
                  <p>Upload a drone frame to mark suspected casualty locations</p>
                </div>
              )}
            </div>
          </div>

          <aside className="manual-points-panel">
            <div className="manual-points-header">
              <strong>Clicked Points</strong>
              <button type="button" onClick={clearManualPoints} disabled={!manualDronePoints.length}>Clear</button>
            </div>
            <div className="manual-point-list">
              {manualDronePoints.length ? manualDronePoints.map((point) => (
                <article key={point.id} className={`manual-point-row ${selectedMarkerId === point.id ? 'is-repositioning' : ''}`}>
                  <div className="manual-point-id">{point.id.replace('manual_', '#')}</div>
                  <div className="manual-point-form">
                    <strong>{point.localization_label}</strong>
                    <label>
                      Soldier ID
                      <input
                        value={point.soldier_id ?? ''}
                        onChange={(event) => updateManualPoint(point.id, { soldier_id: event.target.value })}
                        placeholder="388"
                      />
                    </label>
                    <label>
                      Band ID
                      <input
                        value={point.band_id ?? ''}
                        onChange={(event) => updateManualPoint(point.id, { band_id: event.target.value })}
                        placeholder="RB-388"
                      />
                    </label>
                    <label>
                      Label
                      <input
                        value={point.label ?? 'Soldier'}
                        onChange={(event) => updateManualPoint(point.id, { label: event.target.value })}
                      />
                    </label>
                    <label>
                      Status
                      <select
                        value={point.status ?? ''}
                        onChange={(event) => updateManualPoint(point.id, { status: event.target.value })}
                      >
                        <option value="">Unspecified</option>
                        <option value="Critical">Critical</option>
                        <option value="Urgent">Urgent</option>
                        <option value="Stable">Stable</option>
                      </select>
                    </label>
                    <span>image [{point.image_center.join(', ')}]</span>
                    <span>map [{point.map_position.join(', ')}]</span>
                    <div className="manual-point-actions">
                      <button type="button" onClick={() => setSelectedMarkerId(point.id)}>
                        Reposition
                      </button>
                      <button type="button" onClick={() => deleteManualPoint(point.id)}>
                        Delete
                      </button>
                    </div>
                  </div>
                </article>
              )) : (
                <p>No manual points marked yet.</p>
              )}
            </div>
          </aside>
        </div>
      </section>
        </div>
      </details>
      ) : null}
    </section>
  );
}

export default function App() {
  const [activePage, setActivePage] = useState('mission');
  const [expandedMap, setExpandedMap] = useState(null);
  const [globalYoloOnline, setGlobalYoloOnline] = useState(false);
  const [manualDronePoints, setManualDronePoints] = useState(() => {
    try {
      const stored = window.localStorage.getItem(MANUAL_POINTS_STORAGE_KEY);
      const parsed = stored ? JSON.parse(stored) : [];
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  });
  const cudaData = useCudaData();
  const cudaDataLoaded = Boolean(cudaData.benchmarks?.length && cudaData.riskRanking?.length);

  useEffect(() => {
    let cancelled = false;
    const checkYolo = () => {
      fetch('http://127.0.0.1:8000/api/health')
        .then((response) => {
          if (!cancelled) setGlobalYoloOnline(response.ok);
        })
        .catch(() => {
          if (!cancelled) setGlobalYoloOnline(false);
        });
    };
    checkYolo();
    const interval = setInterval(checkYolo, 5000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  const globalBannerText = cudaDataLoaded
    ? globalYoloOnline
      ? 'Verified CUDA benchmark data loaded · YOLO backend online · live visual detection enabled'
      : 'Verified CUDA benchmark data loaded from Colab export · YOLO backend offline · visual demo fallback active'
    : 'CUDA benchmark data missing · run start_resqvision_demo.ps1 or import Colab outputs';

  useEffect(() => {
    window.localStorage.setItem(MANUAL_POINTS_STORAGE_KEY, JSON.stringify(manualDronePoints));
  }, [manualDronePoints]);

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
    mission: <MissionPlan riskRanking={cudaData.riskRanking} attentionStats={cudaData.attentionStats} fusionMode={cudaData.fusionMode} manualDronePoints={manualDronePoints} setActivePage={setActivePage} setExpandedMap={setExpandedMap} />,
    command: <TacticalCommand riskRanking={cudaData.riskRanking} attentionStats={cudaData.attentionStats} fusionMode={cudaData.fusionMode} manualDronePoints={manualDronePoints} setExpandedMap={setExpandedMap} />,
    analytics: <Analytics benchmarks={cudaData.benchmarks} attentionStats={cudaData.attentionStats} />,
    architecture: <SystemArchitecture />,
    cv: <ComputerVision manualDronePoints={manualDronePoints} setManualDronePoints={setManualDronePoints} refreshTacticalData={cudaData.refresh} />
  }[activePage];

  return (
    <Shell activePage={activePage} setActivePage={setActivePage}>
      <div className="top-strip">
        <div>
          <Activity size={18} />
          <span className="top-strip-message">{globalBannerText}</span>
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
              <TacticalMap planning showArrows soldiers={cudaData.riskRanking ?? topTargets} attentionData={cudaData.attentionStats ?? []} fusionMode={cudaData.fusionMode} manualPoints={manualDronePoints} />
            </div>
          </section>
        </div>
      ) : null}
    </Shell>
  );
}
