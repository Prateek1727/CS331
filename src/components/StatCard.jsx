import { TrendingUp, TrendingDown } from 'lucide-react'
import AnimatedCounter from './AnimatedCounter'

export default function StatCard({ icon: Icon, label, value, change, prefix = '', suffix = '', decimals = 0, gradient = 'var(--gradient-primary)' }) {
  const isPositive = change >= 0

  return (
    <div className="card" style={{ position: 'relative', overflow: 'hidden' }}>
      <div style={{
        position: 'absolute', top: 0, right: 0, width: 80, height: 80,
        background: gradient, opacity: 0.06, borderRadius: '0 0 0 80px'
      }} />
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 12 }}>
        <div style={{
          width: 40, height: 40, borderRadius: 10,
          background: gradient, display: 'flex', alignItems: 'center', justifyContent: 'center',
          boxShadow: '0 4px 12px rgba(6, 182, 212, 0.2)'
        }}>
          <Icon size={20} color="white" />
        </div>
        <div style={{
          display: 'flex', alignItems: 'center', gap: 4,
          padding: '3px 8px', borderRadius: 20,
          background: isPositive ? 'rgba(16,185,129,0.1)' : 'rgba(239,68,68,0.1)',
          fontSize: '0.72rem', fontWeight: 600,
          color: isPositive ? '#34d399' : '#f87171'
        }}>
          {isPositive ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
          {isPositive ? '+' : ''}{change}%
        </div>
      </div>
      <div style={{ fontSize: '1.85rem', fontWeight: 800, letterSpacing: '-0.02em', lineHeight: 1.1 }}>
        <AnimatedCounter end={value} prefix={prefix} suffix={suffix} decimals={decimals} />
      </div>
      <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: 4, fontWeight: 500 }}>
        {label}
      </div>
    </div>
  )
}
