import { useEffect, useRef } from 'react'

export function ScoreProgress({ percent, label }: { percent: number; label: string }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const percentRef = useRef(percent)
  percentRef.current = percent

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const rawCtx = canvas.getContext('2d')
    if (!rawCtx) return
    const ctx: CanvasRenderingContext2D = rawCtx

    const size = canvas.clientWidth
    const dpr = window.devicePixelRatio || 1
    canvas.width = size * dpr
    canvas.height = size * dpr
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

    const cx = size / 2
    const cy = size / 2
    const radius = size / 2 - 8
    let raf = 0
    const start = performance.now()

    function frame(now: number) {
      const t = now - start
      ctx.clearRect(0, 0, size, size)

      ctx.strokeStyle = 'rgba(255,255,255,0.08)'
      ctx.lineWidth = 5
      ctx.beginPath()
      ctx.arc(cx, cy, radius, 0, Math.PI * 2)
      ctx.stroke()

      const frac = Math.min(Math.max(percentRef.current / 100, 0), 1)
      const startAngle = -Math.PI / 2
      const endAngle = startAngle + frac * Math.PI * 2

      const grad = ctx.createLinearGradient(0, 0, size, size)
      grad.addColorStop(0, '#4da6ff')
      grad.addColorStop(1, '#5cff8f')
      ctx.strokeStyle = grad
      ctx.lineWidth = 5
      ctx.lineCap = 'round'
      ctx.beginPath()
      ctx.arc(cx, cy, radius, startAngle, endAngle)
      ctx.stroke()

      const sx = cx + Math.cos(endAngle) * radius
      const sy = cy + Math.sin(endAngle) * radius
      const pulse = 3 + Math.sin(t * 0.01) * 1.5
      ctx.fillStyle = '#ffffff'
      ctx.beginPath()
      ctx.arc(sx, sy, pulse, 0, Math.PI * 2)
      ctx.fill()

      raf = requestAnimationFrame(frame)
    }
    raf = requestAnimationFrame(frame)
    return () => cancelAnimationFrame(raf)
  }, [])

  return (
    <section className="panel progress-panel">
      <div className="progress-ring-wrap">
        <canvas ref={canvasRef} className="progress-ring" />
        <div className="progress-percent">{Math.round(percent)}%</div>
      </div>
      <p className="progress-label">{label}</p>
    </section>
  )
}
