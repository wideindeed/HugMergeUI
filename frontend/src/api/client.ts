import type {
  ArchitectureCheckResult,
  ConflictScoreResult,
  CuratedModel,
  ModelCheckResult,
  ParsedConfig,
  ScoreProgressEvent,
} from './types'

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

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`/api${path}`)
  if (!res.ok) {
    const detail = await res.json().catch(() => null)
    throw new Error(detail?.detail ?? `${path} failed with status ${res.status}`)
  }
  return res.json()
}

export function searchModels(query: string): Promise<string[]> {
  return get(`/hf-search?q=${encodeURIComponent(query)}`)
}

export function checkModel(modelId: string): Promise<ModelCheckResult> {
  return get(`/model-check?model_id=${encodeURIComponent(modelId)}`)
}

export function getCuratedModels(): Promise<CuratedModel[]> {
  return get('/curated-models')
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

export function streamConflictScore(
  baseModel: string,
  modelA: string,
  modelB: string,
  density: number,
  onProgress: (event: ScoreProgressEvent) => void,
): Promise<ConflictScoreResult> {
  const params = new URLSearchParams({
    base_model: baseModel,
    model_a: modelA,
    model_b: modelB,
    density: String(density),
  })

  return new Promise((resolve, reject) => {
    const source = new EventSource(`/api/conflict-score-stream?${params.toString()}`)

    source.onmessage = (raw) => {
      const event = JSON.parse(raw.data) as ScoreProgressEvent
      onProgress(event)
      if (event.stage === 'scored') {
        source.close()
        resolve(event.result)
      } else if (event.stage === 'error') {
        source.close()
        reject(new Error(event.message))
      }
    }

    source.onerror = () => {
      source.close()
      reject(new Error('lost connection to /conflict-score-stream'))
    }
  })
}
