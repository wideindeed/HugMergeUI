import { useEffect, useState, type CSSProperties } from 'react'
import { TOUR_STEPS } from '../data/tourSteps'

interface Rect {
  top: number
  left: number
  width: number
  height: number
}

function findTargetRect(id: string): Rect | null {
  const el = document.querySelector(`[data-tour-id="${id}"]`)
  if (!el) return null
  const r = el.getBoundingClientRect()
  return { top: r.top + window.scrollY, left: r.left + window.scrollX, width: r.width, height: r.height }
}

export function GuideTour({ active, onClose }: { active: boolean; onClose: () => void }) {
  const [stepIndex, setStepIndex] = useState(0)
  const [rect, setRect] = useState<Rect | null>(null)

  const step = TOUR_STEPS[stepIndex]

  useEffect(() => {
    if (active) setStepIndex(0)
  }, [active])

  useEffect(() => {
    if (!active) return

    document.querySelectorAll('.tour-highlight').forEach((el) => el.classList.remove('tour-highlight'))
    const target = document.querySelector(`[data-tour-id="${step.id}"]`)
    target?.classList.add('tour-highlight')
    target?.scrollIntoView({ behavior: 'smooth', block: 'center' })

    const update = () => setRect(findTargetRect(step.id))
    update()
    const timeout = setTimeout(update, 350)
    window.addEventListener('resize', update)
    return () => {
      clearTimeout(timeout)
      window.removeEventListener('resize', update)
      target?.classList.remove('tour-highlight')
    }
  }, [active, step])

  if (!active) return null

  const isLast = stepIndex === TOUR_STEPS.length - 1

  const tooltipStyle: CSSProperties = rect
    ? { position: 'absolute', top: rect.top + rect.height + 12, left: Math.min(rect.left, window.innerWidth - 340) }
    : { position: 'fixed', top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }

  return (
    <>
      <div className="tour-backdrop" onClick={onClose} />
      <div className="tour-tooltip" style={tooltipStyle}>
        <div className="tour-tooltip-header">
          <strong>{step.title}</strong>
          <span className="tour-step-count">
            {stepIndex + 1} / {TOUR_STEPS.length}
          </span>
        </div>
        <p>{step.text}</p>
        <div className="tour-tooltip-actions">
          <button type="button" className="tour-close" onClick={onClose}>
            Close
          </button>
          <div className="tour-tooltip-nav">
            {stepIndex > 0 && (
              <button type="button" onClick={() => setStepIndex((i) => i - 1)}>
                Back
              </button>
            )}
            <button
              type="button"
              className="tour-next"
              onClick={() => (isLast ? onClose() : setStepIndex((i) => i + 1))}
            >
              {isLast ? 'Done' : 'Next'}
            </button>
          </div>
        </div>
      </div>
    </>
  )
}
