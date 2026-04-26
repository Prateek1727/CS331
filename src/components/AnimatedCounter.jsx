import { useEffect, useRef, useState } from 'react'

export default function AnimatedCounter({ end, duration = 1200, prefix = '', suffix = '', decimals = 0 }) {
  const [val, setVal] = useState(0)
  const ref = useRef(null)
  const startTime = useRef(null)

  useEffect(() => {
    const animate = (timestamp) => {
      if (!startTime.current) startTime.current = timestamp
      const progress = Math.min((timestamp - startTime.current) / duration, 1)
      const eased = 1 - Math.pow(1 - progress, 3) // easeOutCubic
      setVal(eased * end)
      if (progress < 1) ref.current = requestAnimationFrame(animate)
    }
    ref.current = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(ref.current)
  }, [end, duration])

  return (
    <span>{prefix}{val.toFixed(decimals)}{suffix}</span>
  )
}
