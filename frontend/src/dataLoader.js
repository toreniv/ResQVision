import { benchmarkRows, correctnessMetrics, topTargets } from './data.js';

export async function loadRiskRanking() {
  // TODO: Fetch a JSON export generated from CUDA risk_ranking.csv / CUDA output artifacts.
  return topTargets;
}

export async function loadBenchmarkResults() {
  // TODO: Fetch benchmark_results.json generated after running the CUDA benchmark workflow.
  return benchmarkRows;
}

export async function loadAttentionStats() {
  // TODO: Fetch attention_stats.json generated from CUDA attention entropy/statistics output.
  return correctnessMetrics;
}
