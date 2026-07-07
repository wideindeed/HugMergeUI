import type { ArchitectureCheckResult, ConflictScoreResult, ParsedConfig } from './types'

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`/api${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const detail = await res.json().catch(() => null)
    throw new Error(detail?.detail ?? `${path} failed with status ${res.status}`)
  }
  return res.json()
}

export function parseConfig(yamlText: string, numLayers: number | null): Promise<ParsedConfig> {
  return post('/parse-config', { yaml_text: yamlText, num_layers: numLayers })
}

export function checkArchitecture(yamlText: string): Promise<ArchitectureCheckResult> {
  return post('/check-architecture', { yaml_text: yamlText })
}

export function conflictScore(
  baseModel: string,
  modelA: string,
  modelB: string,
  density: number,
): Promise<ConflictScoreResult> {
  return post('/conflict-score', { base_model: baseModel, model_a: modelA, model_b: modelB, density })
}
