import { TrendingUp, TrendingDown, AlertTriangle, Brain, RefreshCw, Cpu, Layers } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Cell, RadarChart, PolarGrid, PolarAngleAxis,PolarRadiusAxis, Radar } from 'recharts'
import { useData } from '../context/DataContext'

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload) return null
  return (
    <div style={{
      background: 'rgba(12,17,32,0.95)', backdropFilter: 'blur(10px)',
      border: '1px solid var(--border-primary)', borderRadius: 10,
      padding: '10px 14px', fontSize: '0.75rem',
    }}>
      <div style={{ fontWeight: 600, marginBottom: 6 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
          <div style={{ width: 8, height: 8, borderRadius: 2, background: p.color || p.stroke }} />
          <span style={{ color: 'var(--text-secondary)' }}>{p.name}: </span>
          <span style={{ fontWeight: 600 }}>{typeof p.value === 'number' ? p.value.toFixed(1) : p.value}%</span>
        </div>
      ))}
    </div>
  )
}

export default function Intelligence() {
  const { tickets } = useData()
  const chartData = {
    modelAccuracy: [
      { week: 'W1', nlp: 82, vision: 78, rag: 85, fraud: 80 },
      { week: 'W2', nlp: 85, vision: 81, rag: 88, fraud: 84 },
      { week: 'W3', nlp: 89, vision: 86, rag: 91, fraud: 89 },
      { week: 'W4', nlp: 94, vision: 92, rag: 95, fraud: 93 },
    ],
    topIssues: [
      { category: 'Login Failure', count: 450 },
      { category: 'Billing Dispute', count: 320 },
      { category: 'Missing Package', count: 180 },
      { category: 'Damaged Item', count: 145 },
      { category: 'Account Locked', count: 90 },
    ]
  }
  const issueColors = ['#06b6d4', '#8b5cf6', '#10b981', '#f59e0b', '#ec4899', '#3b82f6', '#f97316', '#64748b']

  return (
    <div className="animate-fade-in">
      <div className="page-header">
        <h1>Intelligence Loop</h1>
        <p>Layer 5 — Continuous learning from every resolved ticket</p>
      </div>

      {/* RLHF Stats */}
      <div style={{ display: 'flex', gap: 16, marginBottom: 24 }}>
        {[
          { icon: Brain, label: 'Model Version', value: 'v3.8.2', color: '#8b5cf6' },
          { icon: Layers, label: 'Training Samples', value: '847K', color: '#06b6d4' },
          { icon: Cpu, label: 'Overall Accuracy', value: '94.2%', color: '#10b981' },
          { icon: RefreshCw, label: 'Last Fine-tuned', value: '2h ago', color: '#f59e0b' },
        ].map(({ icon: Icon, label, value, color }) => (
          <div key={label} className="card" style={{ flex: 1, padding: '16px 18px', display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{
              width: 40, height: 40, borderRadius: 10,
              background: `${color}15`, border: `1px solid ${color}25`,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <Icon size={18} color={color} />
            </div>
            <div>
              <div style={{ fontSize: '1.1rem', fontWeight: 800, color }}>{value}</div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{label}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="grid-2" style={{ marginBottom: 24 }}>
        {/* Model Accuracy Learning Curve */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Model Accuracy — Learning Curve</span>
            <span className="badge badge-success">Improving</span>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={chartData.modelAccuracy}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.06)" />
              <XAxis dataKey="week" tick={{ fill: '#64748b', fontSize: 12 }} axisLine={false} tickLine={false} />
              <YAxis domain={[75, 100]} tick={{ fill: '#64748b', fontSize: 12 }} axisLine={false} tickLine={false} />
              <Tooltip content={<CustomTooltip />} />
              <Line type="monotone" dataKey="nlp" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} name="NLP" />
              <Line type="monotone" dataKey="vision" stroke="#8b5cf6" strokeWidth={2} dot={{ r: 3 }} name="Vision" />
              <Line type="monotone" dataKey="rag" stroke="#06b6d4" strokeWidth={2} dot={{ r: 3 }} name="RAG" />
              <Line type="monotone" dataKey="fraud" stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} name="Fraud" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Top Issues */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Top Issue Categories</span>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={chartData.topIssues} layout="vertical" margin={{ left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.06)" horizontal={false} />
              <XAxis type="number" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis type="category" dataKey="category" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} width={120} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="count" radius={[0, 4, 4, 0]} name="Tickets">
                {chartData.topIssues.map((_, i) => (
                  <Cell key={i} fill={issueColors[i]} fillOpacity={0.8} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid-2">
        {/* Anomaly Alerts */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Anomaly Alerts</span>
            <span className="badge badge-warning"><AlertTriangle size={10} /> 3 Active</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {[
              { title: 'Fraud Spike Detected', desc: 'Image tampering claims up 34% in the last 24 hours', severity: 'critical', time: '2h ago' },
              { title: 'Delivery Issue Surge', desc: 'Delivery complaints trending 18% above baseline this week', severity: 'warning', time: '6h ago' },
              { title: 'Promo Code Abuse', desc: 'Unusual pattern: 12 new accounts using same promo code in 1 hour', severity: 'warning', time: '1d ago' },
            ].map((alert, i) => (
              <div key={i} style={{
                padding: '14px', borderRadius: 10,
                background: alert.severity === 'critical' ? 'rgba(239,68,68,0.06)' : 'rgba(245,158,11,0.04)',
                border: `1px solid ${alert.severity === 'critical' ? 'rgba(239,68,68,0.15)' : 'rgba(245,158,11,0.12)'}`,
              }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <AlertTriangle size={14} color={alert.severity === 'critical' ? '#ef4444' : '#f59e0b'} />
                    <span style={{ fontSize: '0.82rem', fontWeight: 600 }}>{alert.title}</span>
                  </div>
                  <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>{alert.time}</span>
                </div>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>{alert.desc}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Outcome Logger */}
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border-primary)' }}>
            <span className="card-title">Recent Outcomes — Feedback Loop</span>
          </div>
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Ticket</th>
                  <th>AI Prediction</th>
                  <th>Actual Outcome</th>
                  <th>Match</th>
                </tr>
              </thead>
              <tbody>
                {tickets.slice(0, 6).map(t => {
                  const predicted = t.decision.action.replace(/_/g, ' ')
                  const match = t.status === 'auto_resolved' ? t.decision.action === 'auto_resolve' : t.decision.action === 'escalate'
                  return (
                    <tr key={t.id}>
                      <td style={{ fontWeight: 600, color: 'var(--accent-primary)', fontSize: '0.8rem' }}>{t.id}</td>
                      <td style={{ fontSize: '0.78rem', textTransform: 'capitalize' }}>{predicted}</td>
                      <td><span className={`badge ${t.status === 'auto_resolved' ? 'badge-success' : t.status === 'escalated' ? 'badge-danger' : 'badge-warning'}`}>{t.status.replace(/_/g, ' ')}</span></td>
                      <td>
                        {match ? (
                          <span style={{ color: '#10b981', fontWeight: 600, fontSize: '0.78rem' }}>✓ Correct</span>
                        ) : (
                          <span style={{ color: '#f59e0b', fontWeight: 600, fontSize: '0.78rem' }}>△ Learning</span>
                        )}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}
