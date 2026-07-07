import type { ArchitectureWarning } from '../api/types'

function describe(warning: ArchitectureWarning): string {
  if (warning.type === 'config_fetch_failed') {
    return `Could not fetch config.json for ${warning.model}`
  }
  return `Architecture mismatch on ${warning.field}: ${JSON.stringify(warning.values)}`
}

export function ArchitectureWarnings({ warnings }: { warnings: ArchitectureWarning[] }) {
  if (warnings.length === 0) {
    return <p className="ok-banner">Architectures compatible across all referenced models.</p>
  }

  return (
    <ul className="warning-banner">
      {warnings.map((warning, i) => (
        <li key={i}>{describe(warning)}</li>
      ))}
    </ul>
  )
}
