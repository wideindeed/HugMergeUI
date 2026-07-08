import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js'
import type { LayerScore, OtherScore } from '../api/types'

const MAX_RINGS = 16
const CONJUNCTION_WINDOW = 0.2
const FLASH_DURATION = 420
const FLIGHT_DURATION = 900
const COLOR_A = 0x4da6ff
const COLOR_B = 0xffa64d
const OVERVIEW_POS = new THREE.Vector3(14, 6, 14)
const OVERVIEW_TARGET = new THREE.Vector3(0, 0, 0)

interface RingState {
  radius: number
  y: number
  conflict: number
  speedA: number
  speedB: number
  phaseA: number
  phaseB: number
  wasClose: boolean
  flashStart: number | null
  flashColor: THREE.Color
  meshA: THREE.Mesh
  meshB: THREE.Mesh
  pickMesh: THREE.Mesh
  layerLabel: string
  layerCount: number
  tensorCount: number
  redundancyA: number
  redundancyB: number
  drift: number
}

interface Spark {
  mesh: THREE.Mesh
  velocity: THREE.Vector3
  bornAt: number
}

interface SelectedInfo {
  label: string
  isCore: boolean
  layerCount: number
  tensorCount: number
  conflict: number
  redundancyA: number
  redundancyB: number
  drift: number
}

function chunk<T>(arr: T[], parts: number): T[][] {
  if (arr.length <= parts) return arr.map((x) => [x])
  const size = Math.ceil(arr.length / parts)
  const out: T[][] = []
  for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size))
  return out
}

function shapedConflict(conflict: number): number {
  return Math.pow(Math.min(Math.max(conflict, 0), 1), 0.55)
}

function hslColor(conflict: number): THREE.Color {
  const hue = (120 * (1 - shapedConflict(conflict))) / 360
  return new THREE.Color().setHSL(hue, 0.85, 0.55)
}

export function hslCss(conflict: number): string {
  const hue = 120 * (1 - shapedConflict(conflict))
  return `hsl(${hue.toFixed(0)}deg 80% 55%)`
}

function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2
}

export function ConflictScene({ layers, other }: { layers: LayerScore[]; other: OtherScore | null }) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const backRef = useRef<(() => void) | null>(null)
  const [selected, setSelected] = useState<SelectedInfo | null>(null)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const width = container.clientWidth
    const height = container.clientHeight

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 200)
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(width, height)
    renderer.domElement.style.cursor = 'grab'
    container.appendChild(renderer.domElement)

    const composer = new EffectComposer(renderer)
    composer.addPass(new RenderPass(scene, camera))
    const bloom = new UnrealBloomPass(new THREE.Vector2(width, height), 0.55, 0.4, 0.3)
    composer.addPass(bloom)

    camera.position.copy(OVERVIEW_POS)
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.target.copy(OVERVIEW_TARGET)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.autoRotate = true
    controls.autoRotateSpeed = 0.6
    controls.enablePan = false
    controls.minDistance = 3
    controls.maxDistance = 40
    controls.update()

    // starfield
    const starCount = 600
    const starPositions = new Float32Array(starCount * 3)
    for (let i = 0; i < starCount; i++) {
      const r = 25 + Math.random() * 40
      const theta = Math.random() * Math.PI * 2
      const phi = Math.acos(2 * Math.random() - 1)
      starPositions[i * 3] = r * Math.sin(phi) * Math.cos(theta)
      starPositions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta)
      starPositions[i * 3 + 2] = r * Math.cos(phi)
    }
    const starGeo = new THREE.BufferGeometry()
    starGeo.setAttribute('position', new THREE.BufferAttribute(starPositions, 3))
    const stars = new THREE.Points(
      starGeo,
      new THREE.PointsMaterial({ color: 0x99aaff, size: 0.06, transparent: true, opacity: 0.55 }),
    )
    scene.add(stars)

    const bins = chunk(layers, MAX_RINGS)
    const coreConflict = other ? other.conflict : 0

    const coreGeo = new THREE.SphereGeometry(0.9, 32, 32)
    const coreMat = new THREE.MeshBasicMaterial({ color: hslColor(coreConflict) })
    const core = new THREE.Mesh(coreGeo, coreMat)
    scene.add(core)

    const corePickMat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false })
    const corePickMesh = new THREE.Mesh(new THREE.SphereGeometry(1.6, 16, 16), corePickMat)
    scene.add(corePickMesh)

    const ringGroup = new THREE.Group()
    scene.add(ringGroup)

    const ringPickMeshes: THREE.Mesh[] = []

    const rings: RingState[] = bins.map((bin, i) => {
      const avgConflict = bin.reduce((s, l) => s + l.conflict, 0) / bin.length
      const avgRedundancyA = bin.reduce((s, l) => s + l.redundancy_a, 0) / bin.length
      const avgRedundancyB = bin.reduce((s, l) => s + l.redundancy_b, 0) / bin.length
      const avgDrift = bin.reduce((s, l) => s + l.drift_magnitude, 0) / bin.length
      const tensorCount = bin.reduce((s, l) => s + l.tensor_count, 0)
      const layerLabel =
        bin.length === 1 ? `Layer ${bin[0].layer}` : `Layers ${bin[0].layer}–${bin[bin.length - 1].layer}`
      const radius = 2 + (i * 9) / Math.max(bins.length - 1, 1)
      const y = (i / Math.max(bins.length - 1, 1) - 0.5) * 6

      const curve = new THREE.EllipseCurve(0, 0, radius, radius, 0, Math.PI * 2, false, 0)
      const points = curve.getPoints(96).map((p) => new THREE.Vector3(p.x, 0, p.y))
      const ringGeo = new THREE.BufferGeometry().setFromPoints(points)
      const ringMat = new THREE.LineBasicMaterial({
        color: hslColor(avgConflict),
        transparent: true,
        opacity: 0.28,
      })
      const ringLine = new THREE.LineLoop(ringGeo, ringMat)
      ringLine.position.y = y
      ringGroup.add(ringLine)

      const sphereGeo = new THREE.SphereGeometry(0.18, 16, 16)
      const meshA = new THREE.Mesh(sphereGeo, new THREE.MeshBasicMaterial({ color: COLOR_A }))
      const meshB = new THREE.Mesh(sphereGeo.clone(), new THREE.MeshBasicMaterial({ color: COLOR_B }))
      ringGroup.add(meshA)
      ringGroup.add(meshB)

      const pickMesh = new THREE.Mesh(
        new THREE.RingGeometry(Math.max(radius - 0.5, 0.1), radius + 0.5, 64),
        new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide }),
      )
      pickMesh.rotation.x = -Math.PI / 2
      pickMesh.position.y = y
      ringGroup.add(pickMesh)
      ringPickMeshes.push(pickMesh)

      const speed = 0.00034 + 0.00026 * (1 - i / bins.length)
      return {
        radius,
        y,
        conflict: avgConflict,
        speedA: speed,
        speedB: speed * (1 + 0.11 + 0.03 * Math.sin(i)),
        phaseA: Math.random() * Math.PI * 2,
        phaseB: Math.random() * Math.PI * 2,
        wasClose: false,
        flashStart: null,
        flashColor: new THREE.Color(0xffffff),
        meshA,
        meshB,
        pickMesh,
        layerLabel,
        layerCount: bin.length,
        tensorCount,
        redundancyA: avgRedundancyA,
        redundancyB: avgRedundancyB,
        drift: avgDrift,
      }
    })

    const outerRadius = rings.length > 0 ? rings[rings.length - 1].radius : 2
    const gridRing = new THREE.Mesh(
      new THREE.RingGeometry(outerRadius + 0.4, outerRadius + 0.45, 128),
      new THREE.MeshBasicMaterial({ color: 0x335577, transparent: true, opacity: 0.25, side: THREE.DoubleSide }),
    )
    gridRing.rotation.x = Math.PI / 2
    scene.add(gridRing)

    let haloMesh: THREE.Mesh | null = null
    function setHalo(radius: number | null, y: number) {
      if (haloMesh) {
        scene.remove(haloMesh)
        haloMesh.geometry.dispose()
        ;(haloMesh.material as THREE.Material).dispose()
        haloMesh = null
      }
      if (radius === null) return
      const mesh = new THREE.Mesh(
        new THREE.TorusGeometry(radius, 0.045, 8, 96),
        new THREE.MeshBasicMaterial({ color: 0x66ffff, transparent: true, opacity: 0.9 }),
      )
      mesh.rotation.x = Math.PI / 2
      mesh.position.y = y
      scene.add(mesh)
      haloMesh = mesh
    }

    interface Flight {
      fromPos: THREE.Vector3
      toPos: THREE.Vector3
      fromTarget: THREE.Vector3
      toTarget: THREE.Vector3
      start: number
    }
    let flight: Flight | null = null

    function startFlight(toPos: THREE.Vector3, toTarget: THREE.Vector3) {
      flight = {
        fromPos: camera.position.clone(),
        toPos,
        fromTarget: controls.target.clone(),
        toTarget,
        start: performance.now(),
      }
    }

    function focusRing(idx: number) {
      const ring = rings[idx]
      const dist = Math.max(ring.radius * 1.15, 3.5)
      startFlight(
        new THREE.Vector3(dist, ring.y + Math.max(2.2, ring.radius * 0.25), dist),
        new THREE.Vector3(0, ring.y, 0),
      )
      controls.autoRotate = false
      setHalo(ring.radius, ring.y)
      setSelected({
        label: ring.layerLabel,
        isCore: false,
        layerCount: ring.layerCount,
        tensorCount: ring.tensorCount,
        conflict: ring.conflict,
        redundancyA: ring.redundancyA,
        redundancyB: ring.redundancyB,
        drift: ring.drift,
      })
    }

    function focusCore() {
      startFlight(new THREE.Vector3(3.4, 2.2, 3.4), new THREE.Vector3(0, 0, 0))
      controls.autoRotate = false
      setHalo(null, 0)
      setSelected({
        label: 'Non-layer tensors',
        isCore: true,
        layerCount: 0,
        tensorCount: other?.tensor_count ?? 0,
        conflict: coreConflict,
        redundancyA: other?.redundancy_a ?? 0,
        redundancyB: other?.redundancy_b ?? 0,
        drift: other?.drift_magnitude ?? 0,
      })
    }

    function resetView() {
      startFlight(OVERVIEW_POS.clone(), OVERVIEW_TARGET.clone())
      controls.autoRotate = true
      setHalo(null, 0)
      setSelected(null)
    }
    backRef.current = resetView

    const pickTargets: THREE.Object3D[] = [corePickMesh, ...ringPickMeshes]
    const raycaster = new THREE.Raycaster()
    let downX = 0
    let downY = 0
    let downTime = 0

    function pointerToNdc(e: PointerEvent): THREE.Vector2 {
      const rect = renderer.domElement.getBoundingClientRect()
      return new THREE.Vector2(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1,
      )
    }

    function onPointerDown(e: PointerEvent) {
      downX = e.clientX
      downY = e.clientY
      downTime = performance.now()
    }

    function onPointerUp(e: PointerEvent) {
      const dx = e.clientX - downX
      const dy = e.clientY - downY
      const dt = performance.now() - downTime
      if (Math.hypot(dx, dy) > 6 || dt > 500) return
      raycaster.setFromCamera(pointerToNdc(e), camera)
      const hits = raycaster.intersectObjects(pickTargets, false)
      if (hits.length === 0) return
      const hit = hits[0].object
      if (hit === corePickMesh) {
        focusCore()
        return
      }
      const idx = ringPickMeshes.indexOf(hit as THREE.Mesh)
      if (idx >= 0) focusRing(idx)
    }

    function onPointerMove(e: PointerEvent) {
      raycaster.setFromCamera(pointerToNdc(e), camera)
      const hits = raycaster.intersectObjects(pickTargets, false)
      renderer.domElement.style.cursor = hits.length > 0 ? 'pointer' : 'grab'
    }

    renderer.domElement.addEventListener('pointerdown', onPointerDown)
    renderer.domElement.addEventListener('pointerup', onPointerUp)
    renderer.domElement.addEventListener('pointermove', onPointerMove)

    const sparks: Spark[] = []
    const sparkGeo = new THREE.SphereGeometry(0.07, 8, 8)

    function normAngleDiff(a: number, b: number): number {
      let d = Math.abs(a - b) % (Math.PI * 2)
      if (d > Math.PI) d = Math.PI * 2 - d
      return d
    }

    let raf = 0
    const startTime = performance.now()

    function frame(now: number) {
      const t = now - startTime

      controls.update()

      if (flight) {
        const ft = Math.min((now - flight.start) / FLIGHT_DURATION, 1)
        const eased = easeInOutCubic(ft)
        camera.position.lerpVectors(flight.fromPos, flight.toPos, eased)
        controls.target.lerpVectors(flight.fromTarget, flight.toTarget, eased)
        if (ft >= 1) flight = null
      }

      core.scale.setScalar(1 + 0.08 * Math.sin(t * 0.004))
      ringGroup.rotation.y = t * 0.00004
      stars.rotation.y = t * 0.00001
      gridRing.rotation.z = -t * 0.00002
      if (haloMesh) {
        const mat = haloMesh.material as THREE.MeshBasicMaterial
        mat.opacity = 0.55 + 0.35 * Math.sin(t * 0.006)
      }

      for (const ring of rings) {
        const angleA = ring.phaseA + t * ring.speedA
        const angleB = ring.phaseB + t * ring.speedB

        ring.meshA.position.set(Math.cos(angleA) * ring.radius, ring.y, Math.sin(angleA) * ring.radius)
        ring.meshB.position.set(Math.cos(angleB) * ring.radius, ring.y, Math.sin(angleB) * ring.radius)

        const close = normAngleDiff(angleA, angleB) < CONJUNCTION_WINDOW
        if (close && !ring.wasClose) {
          const color = ring.conflict > 0.5 ? 0xff5c5c : 0x5cff8f
          ring.flashStart = t
          ring.flashColor = new THREE.Color(color)

          const mid = ring.meshA.position.clone().add(ring.meshB.position).multiplyScalar(0.5)
          const count = 6 + Math.round(ring.conflict * 8)
          for (let i = 0; i < count; i++) {
            const mesh = new THREE.Mesh(sparkGeo, new THREE.MeshBasicMaterial({ color, transparent: true }))
            mesh.position.copy(mid)
            const dir = new THREE.Vector3(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5).normalize()
            const speed = 0.004 + Math.random() * 0.006
            ringGroup.add(mesh)
            sparks.push({ mesh, velocity: dir.multiplyScalar(speed), bornAt: t })
          }
        }
        ring.wasClose = close

        if (ring.flashStart !== null) {
          const age = t - ring.flashStart
          if (age > FLASH_DURATION) {
            ring.flashStart = null
            ;(ring.meshA.material as THREE.MeshBasicMaterial).color.setHex(COLOR_A)
            ;(ring.meshB.material as THREE.MeshBasicMaterial).color.setHex(COLOR_B)
            ring.meshA.scale.setScalar(1)
            ring.meshB.scale.setScalar(1)
          } else {
            const pulse = Math.sin((age / FLASH_DURATION) * Math.PI)
            const scale = 1 + pulse * 1.6
            ring.meshA.scale.setScalar(scale)
            ring.meshB.scale.setScalar(scale)
            ;(ring.meshA.material as THREE.MeshBasicMaterial).color.setHex(COLOR_A).lerp(ring.flashColor, pulse)
            ;(ring.meshB.material as THREE.MeshBasicMaterial).color.setHex(COLOR_B).lerp(ring.flashColor, pulse)
          }
        }
      }

      for (let i = sparks.length - 1; i >= 0; i--) {
        const s = sparks[i]
        const age = t - s.bornAt
        if (age > 700) {
          ringGroup.remove(s.mesh)
          sparks.splice(i, 1)
          continue
        }
        s.mesh.position.addScaledVector(s.velocity, 16)
        const mat = s.mesh.material as THREE.MeshBasicMaterial
        mat.opacity = 1 - age / 700
      }

      composer.render()
      raf = requestAnimationFrame(frame)
    }

    raf = requestAnimationFrame(frame)

    const resizeObserver = new ResizeObserver(() => {
      const w = container.clientWidth
      const h = container.clientHeight
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
      composer.setSize(w, h)
    })
    resizeObserver.observe(container)

    return () => {
      cancelAnimationFrame(raf)
      resizeObserver.disconnect()
      controls.dispose()
      renderer.domElement.removeEventListener('pointerdown', onPointerDown)
      renderer.domElement.removeEventListener('pointerup', onPointerUp)
      renderer.domElement.removeEventListener('pointermove', onPointerMove)
      backRef.current = null
      container.removeChild(renderer.domElement)
      scene.traverse((obj) => {
        if (obj instanceof THREE.Mesh || obj instanceof THREE.LineLoop || obj instanceof THREE.Points) {
          obj.geometry.dispose()
          const mat = obj.material
          if (Array.isArray(mat)) mat.forEach((m) => m.dispose())
          else mat.dispose()
        }
      })
      renderer.dispose()
    }
  }, [layers, other])

  return (
    <section className="panel orbit-panel">
      <h2>Merge interaction</h2>
      <div ref={containerRef} className="orbit-canvas-3d">
        {selected && (
          <div className="inspect-panel">
            <h3>{selected.label}</h3>
            <div className="inspect-row">
              <div className="inspect-row-label">
                <span>Conflict</span>
                <span>{Math.round(selected.conflict * 100)}%</span>
              </div>
              <div className="inspect-bar">
                <div
                  className="inspect-bar-fill"
                  style={{ width: `${Math.round(selected.conflict * 100)}%`, background: hslCss(selected.conflict) }}
                />
              </div>
            </div>
            <div className="inspect-row">
              <div className="inspect-row-label">
                <span>Redundancy A</span>
                <span>{Math.round(selected.redundancyA * 100)}%</span>
              </div>
              <div className="inspect-bar">
                <div
                  className="inspect-bar-fill"
                  style={{ width: `${Math.round(selected.redundancyA * 100)}%`, background: '#4da6ff' }}
                />
              </div>
            </div>
            <div className="inspect-row">
              <div className="inspect-row-label">
                <span>Redundancy B</span>
                <span>{Math.round(selected.redundancyB * 100)}%</span>
              </div>
              <div className="inspect-bar">
                <div
                  className="inspect-bar-fill"
                  style={{ width: `${Math.round(selected.redundancyB * 100)}%`, background: '#ffa64d' }}
                />
              </div>
            </div>
            <p className="inspect-meta">
              Drift magnitude {selected.drift.toFixed(3)}
              {!selected.isCore && ` · ${selected.layerCount} layer${selected.layerCount === 1 ? '' : 's'}`}
              {' · '}
              {selected.tensorCount} tensor{selected.tensorCount === 1 ? '' : 's'}
            </p>
            <button type="button" className="inspect-back" onClick={() => backRef.current?.()}>
              ← Back to overview
            </button>
          </div>
        )}
      </div>
      <div className="orbit-legend">
        <p className="orbit-legend-intro">
          Drag to orbit, scroll to zoom — it also drifts on its own. Click any ring or the center sphere to zoom in
          and inspect its real data.
        </p>
        <dl className="orbit-legend-grid">
          <div>
            <dt>Center</dt>
            <dd>Non-layer tensors — embeddings, norms, lm_head.</dd>
          </div>
          <div>
            <dt>Rings</dt>
            <dd>Each ring is a layer (or a band of layers for deep models), stacked in a spiral so rotating reveals real depth.</dd>
          </div>
          <div>
            <dt>Blue / orange</dt>
            <dd>Model A and model B, orbiting at slightly different speeds so they drift in and out of alignment.</dd>
          </div>
          <div>
            <dt>Flashes</dt>
            <dd>Green = sign agreement, red = conflict, right on the orbiting pair. Size and color track that layer's real conflict score.</dd>
          </div>
        </dl>
      </div>
    </section>
  )
}
