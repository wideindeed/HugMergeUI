import type { ArchitectureWarning } from '../api/types'
import { useSimpleMode } from '../context/SimpleModeContext'

function describe(warning: ArchitectureWarning): string {
  if (warning.type === 'config_fetch_failed') {
    return `Could not fetch config.json for ${warning.model}`
  }
  return `Architecture mismatch on ${warning.field}: ${JSON.stringify(warning.values)}`
}

function describeSimple(warning: ArchitectureWarning): string {
  if (warning.type === 'config_fetch_failed') {
    return `Couldn't even find the blueprint for ${warning.model} — that one might not be a real, public model.`
  }
  return `These two models are built differently under the hood (mismatched "${warning.field}") — like trying to bolt together parts from two different machines.`
}

export function ArchitectureWarnings({ warnings }: { warnings: ArchitectureWarning[] }) {
  const { simple } = useSimpleMode()

  if (warnings.length === 0) {
    return (
      <p className="ok-banner">
        {simple
          ? "Good news — these models are built the same way underneath. They're compatible."
          : 'Architectures compatible across all referenced models.'}
      </p>
    )
  }

  return (
    <ul className="warning-banner">
      {warnings.map((warning, i) => (
        <li key={i}>{simple ? describeSimple(warning) : describe(warning)}</li>
      ))}
    </ul>
  )
}
