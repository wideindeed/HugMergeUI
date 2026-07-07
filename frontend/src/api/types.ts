export interface ParsedConfig {
  merge_method: string
  base_model: string | null
  dtype: string | null
  models: string[]
}

export type ArchitectureWarning =
  | { type: 'config_fetch_failed'; model: string }
  | { type: 'architecture_mismatch'; field: string; values: Record<string, unknown> }

export interface ArchitectureCheckResult {
  models: Record<string, Record<string, unknown> | null>
  warnings: ArchitectureWarning[]
  num_layers: number
}

export interface LayerScore {
  layer: number
  tensor_count: number
  conflict: number
  redundancy_a: number
  redundancy_b: number
}

export interface OtherScore {
  tensor_count: number
  conflict: number
  redundancy_a: number
  redundancy_b: number
}

export interface ConflictScoreResult {
  layers: LayerScore[]
  other: OtherScore | null
}
