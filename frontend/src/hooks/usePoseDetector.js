import { useEffect, useRef, useState, useCallback } from 'react'

/**
 * usePoseDetector
 * ---------------
 * Loads TF.js MoveNet (Thunder) using the globally available tf and
 * poseDetection objects injected via CDN <script> tags in index.html.
 *
 * Using CDN globals bypasses Vite/Rollup's bundler, which was producing
 * a "Cannot access before initialization" error due to circular dependencies
 * between @tensorflow/tfjs-core and @tensorflow/tfjs-backend-webgl.
 *
 * Latency: ~10-20ms on WebGL vs 100-300ms YOLO on a CPU-only VPS.
 */
export function usePoseDetector() {
  const detectorRef = useRef(null)
  const [isReady, setIsReady] = useState(false)

  useEffect(() => {
    let cancelled = false

    async function load() {
      try {
        const tf            = window.tf
        const poseDetection = window.poseDetection

        if (!tf || !poseDetection) {
          // CDN scripts haven't loaded yet — retry in 200ms
          setTimeout(load, 200)
          return
        }

        // Use WebGL for GPU-accelerated inference; fall back to CPU if unavailable
        try {
          await tf.setBackend('webgl')
          await tf.ready()
        } catch {
          await tf.setBackend('cpu')
          await tf.ready()
        }

        const detector = await poseDetection.createDetector(
          poseDetection.SupportedModels.MoveNet,
          { modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER }
        )

        if (!cancelled) {
          detectorRef.current = detector
          setIsReady(true)
        }
      } catch (err) {
        console.error('[usePoseDetector] Failed to load model:', err)
      }
    }

    load()
    return () => { cancelled = true }
  }, [])

  /**
   * detectPose(videoElement)
   * Returns [[x, y, score], ...] for 17 COCO keypoints in pixel coords.
   * Returns null if model isn't ready or no pose is found.
   */
  const detectPose = useCallback(async (videoEl) => {
    if (!detectorRef.current || !videoEl || videoEl.readyState < 2) return null

    const poses = await detectorRef.current.estimatePoses(videoEl)
    if (!poses.length) return null

    // Gate on per-joint torso confidence (shoulders + hips).
    // Using a minimum (not average): if ANY torso joint is uncertain the whole
    // pose is unreliable. Average lets a strong shoulder compensate for a
    // missing hip, which causes jitter when only half the body is in frame.
    const KEY_IDXS  = [5, 6, 11, 12]   // L/R shoulder, L/R hip
    const keyScores = KEY_IDXS.map(i => poses[0].keypoints[i]?.score ?? 0)
    if (keyScores.some(s => s < 0.35)) return null

    return poses[0].keypoints.map(kp => [kp.x, kp.y, kp.score ?? 0])
  }, [])

  return { detectPose, isReady }
}
