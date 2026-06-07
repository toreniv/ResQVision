// Normalize helpers - handle multiple possible column name variants
function normalizeBenchmarkRow(row) {
  const soldiers = Number(row.N ?? row.soldiers ?? row.n ?? 0);
  const cpu = Number(row.CPU_time_ms ?? row.cpu ?? row.cpu_ms ?? 0);
  const gpu = Number(row.GPU_time_ms ?? row.gpu ?? row.gpu_ms ?? 0);
  const speedup = Number(row.speedup ?? row.Speedup ?? 0);
  if (!soldiers || !cpu) return null;
  return {
    soldiers,
    cpu,
    gpu,
    speedup,
    correctness: row.correctness ?? row.status,
    maxAbsError: row.max_abs_error ?? row.maxAbsError,
    meanAbsError: row.mean_abs_error ?? row.meanAbsError,
    top10Overlap: row.top10_overlap ?? row.top10Overlap
  };
}

function normalizeRiskRow(row, index) {
  const id = row.soldier_id ?? row.id ?? row.SoldierID ?? `S-${index}`;
  const rawRiskValue = row.risk_score ?? row.risk ?? row.risk_percent;
  const hasRisk = rawRiskValue !== undefined && rawRiskValue !== null && rawRiskValue !== '';
  const riskRaw = Number(hasRisk ? rawRiskValue : 0);
  const risk = hasRisk ? (riskRaw > 1 ? riskRaw / 100 : riskRaw) : null;
  const hr = Number(row.heart_rate ?? row.hr ?? row.hr_bpm ?? 0);
  const spo2 = Number(row.spo2 ?? row.SpO2 ?? row.spo2_percent ?? 0);
  const category = (row.category ?? row.Category ?? (risk >= 0.8 ? 'critical' : risk >= 0.6 ? 'urgent' : 'stable')).toLowerCase();
  const x = Number(row.x_map ?? row.map_x ?? row.x_position ?? row.x ?? 500);
  const y = Number(row.y_map ?? row.map_y ?? row.y_position ?? row.y ?? 500);
  if (!id || (risk === 0 && row.source !== 'MANUAL_TAG')) return null;
  return {
    rank: row.priority ?? row.rank ?? index + 1,
    id,
    soldierId: row.soldier_id ?? id,
    bandId: row.band_id ?? row.bandId,
    risk,
    hr,
    spo2,
    category,
    x,
    y,
    confidence: row.confidence,
    recommendedAction: row.recommended_action ?? row.recommendedAction,
    source: row.source,
    imageCenter: row.image_center ?? row.imageCenter,
    mapPosition: row.map_position ?? row.mapPosition,
    localizationMode: row.localization_mode ?? row.localizationMode,
    localizationLabel: row.localization_label ?? row.localizationLabel,
    telemetryStatus: row.telemetry_status ?? row.telemetryStatus,
    label: row.label,
    status: row.status,
  };
}

// NOTE: correctness fields may live in benchmark_results, not attention_stats.
// normalizeAttentionRow accepts a row from either file.
function normalizeAttentionRow(row) {
  return {
    status: row.correctness ?? row.status ?? 'PASS',
    top10Overlap: row.top10_overlap ?? row.top10Overlap ?? '10/10',
    maxAbsError: row.max_abs_error ?? row.maxAbsError ?? 'N/A',
    meanAbsError: row.mean_abs_error ?? row.meanAbsError ?? 'N/A'
  };
}

async function safeFetch(url) {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    const data = await res.json();
    const rows = Array.isArray(data) ? data : data?.targets;
    console.log(`[CUDA JSON] loaded ${url}: ${rows?.length ?? 0} rows, columns:`, Object.keys(rows?.[0] ?? data ?? {}));
    return data;
  } catch (e) {
    console.warn(`[CUDA JSON] failed to load ${url}:`, e.message);
    return null;
  }
}

export async function loadBenchmarkResults() {
  const raw = await safeFetch('/data/benchmark_results.json');
  if (!raw) return null;
  return raw.map(normalizeBenchmarkRow).filter(Boolean);
}

export async function loadRiskRanking() {
  const fusion = await safeFetch('/data/tactical_fusion.json');
  if (fusion?.fusion_mode !== 'NO_DATA' && Array.isArray(fusion?.targets) && fusion.targets.length) {
    const targets = fusion.targets.map(normalizeRiskRow).filter(Boolean);
    targets.fusionMode = fusion.fusion_mode;
    return targets;
  }

  const raw = await safeFetch('/data/risk_ranking.json');
  if (!raw) return null;
  const targets = raw.map(normalizeRiskRow).filter(Boolean);
  targets.fusionMode = fusion?.fusion_mode ?? null;
  return targets;
}

export async function loadAttentionStats() {
  // Try attention_stats first, fallback to benchmark_results for correctness fields
  const raw = await safeFetch('/data/attention_stats.json')
    ?? await safeFetch('/data/benchmark_results.json');
  if (!raw || !raw.length) return null;
  // Use last row of benchmark (N=1024) for correctness metrics if available
  const row = raw[raw.length - 1];
  return normalizeAttentionRow(row);
}
