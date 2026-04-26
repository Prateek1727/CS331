import { useEffect, useState } from 'react'

export default function TrustScoreGauge({ score, size = 120 }) {
  const [animatedScore, setAnimatedScore] = useState(0)
  const radius = (size - 16) / 2
  const circumference = 2 * Math.PI * radius
  const strokeDashoffset = circumference - (animatedScore / 100) * circumference

  useEffect(() => {
    const timer = setTimeout(() => setAnimatedScore(score), 100)
    return () => clearTimeout(timer)
  }, [score])

  const getColor = (s) => {
    if (s >= 80) return '#10b981'
    if (s >= 50) return '#f59e0b'
    return '#ef4444'
  }
  const getLabel = (s) => {
    if (s >= 80) return 'Trusted'
    if (s >= 50) return 'Medium'
    return 'High Risk'
  }
  const color = getColor(score)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
      <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
        <circle
          cx={size / 2} cy={size / 2} r={radius}
          stroke="rgba(148,163,184,0.1)" strokeWidth="8" fill="none"
        />
        <circle
          cx={size / 2} cy={size / 2} r={radius}
          stroke={color} strokeWidth="8" fill="none"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          style={{ transition: 'stroke-dashoffset 1.2s cubic-bezier(0.4, 0, 0.2, 1), stroke 0.3s ease' }}
        />
        <text
          x={size / 2} y={size / 2}
          textAnchor="middle" dominantBaseline="central"
          style={{
            transform: 'rotate(90deg)',
            transformOrigin: `${size / 2}px ${size / 2}px`,
            fill: color,
            fontSize: size * 0.24,
            fontWeight: 700,
            fontFamily: 'Inter',
          }}
        >
          {score}
        </text>
      </svg>
      <span style={{ fontSize: '0.75rem', color, fontWeight: 600, letterSpacing: '0.05em', textTransform: 'uppercase' }}>
        {getLabel(score)}
      </span>
    </div>
  )
}
