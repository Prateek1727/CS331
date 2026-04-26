import { useState } from 'react'
import { GitBranch, ToggleLeft, ToggleRight, ArrowRight, Crown, Scale, AlertTriangle, CheckCircle2, Clock } from 'lucide-react'
import { useData } from '../context/DataContext'

const RuleCard = ({ rule }) => {
  const [enabled, setEnabled] = useState(rule.enabled)
  const catColors = {
    refund: '#06b6d4', routing: '#8b5cf6', fraud: '#ef4444',
    sla: '#f59e0b', billing: '#3b82f6', compliance: '#64748b',
  }
  const color = catColors[rule.category] || '#64748b'

  return (
    <div className="card" style={{ padding: '18px 20px' }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 8, height: 8, borderRadius: '50%', background: color,
            boxShadow: `0 0 8px ${color}50`,
          }} />
          <span style={{ fontSize: '0.88rem', fontWeight: 600 }}>{rule.name}</span>
        </div>
        <button onClick={() => setEnabled(!enabled)} style={{ color: enabled ? '#10b981' : 'var(--text-muted)', transition: 'color 0.2s' }}>
          {enabled ? <ToggleRight size={24} /> : <ToggleLeft size={24} />}
        </button>
      </div>
      <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: 10 }}>
        {rule.description}
      </p>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ fontSize: '0.68rem', padding: '2px 8px', borderRadius: 20, background: `${color}15`, color, border: `1px solid ${color}25`, fontWeight: 500 }}>
          {rule.category}
        </span>
        {rule.threshold && (
          <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>
            Threshold: <strong style={{ color: 'var(--text-primary)' }}>{rule.threshold}</strong>
          </span>
        )}
        <span style={{ fontSize: '0.62rem', color: 'var(--text-muted)', marginLeft: 'auto' }}>{rule.id}</span>
      </div>
    </div>
  )
}

export default function DecisionEngine() {
  const { tickets, businessRules } = useData()
  
  const recentDecisions = tickets.slice(0, 8).map(t => ({
    ...t,
    decisionType: t.decision.action,
    confidence: t.decision.confidence,
    rule: t.decision.rule,
  }))

  return (
    <div className="animate-fade-in">
      <div className="page-header">
        <h1>Decision Engine</h1>
        <p>Layer 3 — Rules + AI reasoning determine the right action for every ticket</p>
      </div>

      {/* Confidence Threshold Visualizer */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-header">
          <span className="card-title">Confidence Threshold Zones</span>
        </div>
        <div style={{ position: 'relative', height: 60, borderRadius: 10, overflow: 'hidden', background: 'rgba(15,23,42,0.5)' }}>
          <div style={{ position: 'absolute', left: 0, top: 0, height: '100%', width: '40%', background: 'linear-gradient(90deg, rgba(239,68,68,0.15), rgba(245,158,11,0.15))', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#f87171' }}>Escalate to Human</span>
          </div>
          <div style={{ position: 'absolute', left: '40%', top: 0, height: '100%', width: '30%', background: 'rgba(245,158,11,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', borderLeft: '2px dashed rgba(245,158,11,0.3)', borderRight: '2px dashed rgba(6,182,212,0.3)' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#fbbf24' }}>Draft + Review</span>
          </div>
          <div style={{ position: 'absolute', left: '70%', top: 0, height: '100%', width: '30%', background: 'linear-gradient(90deg, rgba(6,182,212,0.1), rgba(16,185,129,0.15))', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#34d399' }}>Auto-Resolve</span>
          </div>
          {/* Markers */}
          <div style={{ position: 'absolute', left: '40%', top: -2, bottom: -2, width: 2 }}>
            <div style={{ position: 'absolute', top: -16, left: -8, fontSize: '0.62rem', color: 'var(--text-muted)' }}>0.70</div>
          </div>
          <div style={{ position: 'absolute', left: '70%', top: -2, bottom: -2, width: 2 }}>
            <div style={{ position: 'absolute', top: -16, left: -8, fontSize: '0.62rem', color: 'var(--text-muted)' }}>0.85</div>
          </div>
        </div>
      </div>

      <div className="grid-2" style={{ marginBottom: 24 }}>
        {/* Escalation Router */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Escalation Router</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {[
              { icon: Crown, label: 'VIP Customer', target: 'Specialist Queue', color: '#f59e0b' },
              { icon: Scale, label: 'Legal Risk', target: 'Compliance Team', color: '#8b5cf6' },
              { icon: AlertTriangle, label: 'Fraud Detected', target: 'Fraud Investigation', color: '#ef4444' },
              { icon: Clock, label: 'SLA Breach', target: 'Priority Escalation', color: '#3b82f6' },
            ].map(({ icon: Icon, label, target, color }) => (
              <div key={label} style={{
                display: 'flex', alignItems: 'center', gap: 12,
                padding: '10px 14px', borderRadius: 10,
                background: `${color}08`, border: `1px solid ${color}15`,
              }}>
                <Icon size={16} color={color} />
                <span style={{ fontSize: '0.82rem', fontWeight: 500 }}>{label}</span>
                <ArrowRight size={14} color="var(--text-muted)" style={{ marginLeft: 'auto' }} />
                <span style={{ fontSize: '0.78rem', color, fontWeight: 500 }}>{target}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Quick Stats */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Decision Stats — Today</span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            {[
              { label: 'Auto-Resolved', value: '247', color: '#10b981', icon: CheckCircle2 },
              { label: 'Draft + Review', value: '58', color: '#f59e0b', icon: Clock },
              { label: 'Escalated', value: '31', color: '#ef4444', icon: AlertTriangle },
              { label: 'Avg Confidence', value: '0.89', color: '#06b6d4', icon: GitBranch },
            ].map(({ label, value, color, icon: Icon }) => (
              <div key={label} style={{
                padding: '14px', borderRadius: 10,
                background: `${color}08`, border: `1px solid ${color}12`,
                textAlign: 'center',
              }}>
                <Icon size={20} color={color} style={{ marginBottom: 6 }} />
                <div style={{ fontSize: '1.3rem', fontWeight: 800, color }}>{value}</div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Business Rules */}
      <div className="card-header" style={{ marginBottom: 16 }}>
        <span className="card-title">Business Rules — Guardrails</span>
      </div>
      <div className="grid-2" style={{ marginBottom: 24 }}>
        {businessRules.map(rule => (
          <RuleCard key={rule.id} rule={rule} />
        ))}
      </div>

      {/* Recent Decisions Log */}
      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border-primary)' }}>
          <span className="card-title">Recent Decisions</span>
        </div>
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Ticket</th>
                <th>Decision</th>
                <th>Confidence</th>
                <th>Rule Applied</th>
              </tr>
            </thead>
            <tbody>
              {recentDecisions.map(d => (
                <tr key={d.id}>
                  <td style={{ fontWeight: 600, color: 'var(--accent-primary)', fontSize: '0.8rem' }}>{d.id}</td>
                  <td>
                    <span className={`badge ${d.decisionType === 'auto_resolve' ? 'badge-success' : d.decisionType === 'escalate' ? 'badge-danger' : 'badge-warning'}`}>
                      {d.decisionType.replace(/_/g, ' ')}
                    </span>
                  </td>
                  <td>
                    <span style={{ fontWeight: 700, color: d.confidence > 0.85 ? '#10b981' : d.confidence > 0.7 ? '#f59e0b' : '#ef4444' }}>
                      {(d.confidence * 100).toFixed(0)}%
                    </span>
                  </td>
                  <td style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', maxWidth: 300, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {d.rule}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
