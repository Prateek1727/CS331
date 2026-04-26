import { Ticket, CheckCircle2, Clock, AlertTriangle, ShieldCheck, DollarSign } from 'lucide-react'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line } from 'recharts'
import StatCard from '../components/StatCard'
import ChannelIcon from '../components/ChannelIcon'
import StatusBadge from '../components/StatusBadge'
import { useData } from '../context/DataContext'
import { CHANNELS, SENTIMENTS } from '../data/constants'
import { useNavigate } from 'react-router-dom'

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
          <div style={{ width: 8, height: 8, borderRadius: 2, background: p.color }} />
          <span style={{ color: 'var(--text-secondary)' }}>{p.name}: </span>
          <span style={{ fontWeight: 600 }}>{p.value}</span>
        </div>
      ))}
    </div>
  )
}

export default function Dashboard() {
  const navigate = useNavigate()
  const { tickets } = useData()

  const len = tickets.length
  const kpiData = {
    totalTickets: { value: len, change: '+0%' },
    autoResolved: { value: len ? (tickets.filter(t => t.status === 'auto_resolved').length / len * 100) : 0, change: '+0%' },
    avgResponseTime: { value: 12.4, change: '-2%' },
    csatScore: { value: 4.8, change: '+0.1' },
  }

  const chartData = {
    ticketVolume: [
      { day: 'Mon', email: 45, chat: 80, voice: 12, social: 25 },
      { day: 'Tue', email: 52, chat: 85, voice: 15, social: 28 },
      { day: 'Wed', email: 48, chat: 92, voice: 18, social: 32 },
      { day: 'Thu', email: 61, chat: 110, voice: 24, social: 45 },
      { day: 'Fri', email: 55, chat: 95, voice: 14, social: 30 },
      { day: 'Sat', email: 30, chat: 45, voice: 8, social: 40 },
      { day: 'Sun', email: 80 + len, chat: 130 + len, voice: 11, social: 35 },
    ],
    resolutionBreakdown: [
      { name: 'Auto-Resolved', value: len ? Math.round(tickets.filter(t => t.status === 'auto_resolved').length / len * 100) : 60, fill: '#10b981' },
      { name: 'Drafted', value: len ? Math.round(tickets.filter(t => t.decision?.action === 'draft_response').length / len * 100) : 25, fill: '#f59e0b' },
      { name: 'Escalated', value: len ? Math.round(tickets.filter(t => t.decision?.action === 'escalate').length / len * 100) : 15, fill: '#ef4444' },
    ]
  }

  return (
    <div className="animate-fade-in">
      <div className="page-header">
        <h1>Command Center</h1>
        <p>Real-time overview of your AI support operations</p>
      </div>

      {/* KPI Cards */}
      <div className="grid-4" style={{ marginBottom: 24 }}>
        <StatCard icon={Ticket} label="Total Tickets (This Week)" value={kpiData.totalTickets.value} change={kpiData.totalTickets.change} />
        <StatCard icon={CheckCircle2} label="Auto-Resolved Rate" value={kpiData.autoResolved.value} suffix="%" decimals={1} change={kpiData.autoResolved.change} gradient="var(--gradient-success)" />
        <StatCard icon={Clock} label="Avg Response Time" value={kpiData.avgResponseTime.value} suffix="s" change={kpiData.avgResponseTime.change} gradient="var(--gradient-accent)" />
        <StatCard icon={ShieldCheck} label="CSAT Score" value={kpiData.csatScore.value} suffix="/5" decimals={2} change={kpiData.csatScore.change} gradient="var(--gradient-warm)" />
      </div>

      {/* Charts Row */}
      <div className="grid-2" style={{ marginBottom: 24 }}>
        {/* Ticket Volume */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Ticket Volume — 7 Day Trend</span>
          </div>
          <ResponsiveContainer width="100%" height={240}>
            <AreaChart data={chartData.ticketVolume}>
              <defs>
                <linearGradient id="colorEmail" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="colorChat" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.06)" />
              <XAxis dataKey="day" tick={{ fill: '#64748b', fontSize: 12 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: '#64748b', fontSize: 12 }} axisLine={false} tickLine={false} />
              <Tooltip content={<CustomTooltip />} />
              <Area type="monotone" dataKey="chat" stroke="#10b981" fill="url(#colorChat)" strokeWidth={2} name="Chat" />
              <Area type="monotone" dataKey="email" stroke="#3b82f6" fill="url(#colorEmail)" strokeWidth={2} name="Email" />
              <Area type="monotone" dataKey="voice" stroke="#f59e0b" fill="none" strokeWidth={1.5} name="Voice" strokeDasharray="4 4" />
              <Area type="monotone" dataKey="social" stroke="#ec4899" fill="none" strokeWidth={1.5} name="Social" />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Resolution Breakdown */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Resolution Breakdown</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
            <ResponsiveContainer width="50%" height={240}>
              <PieChart>
                <Pie
                  data={chartData.resolutionBreakdown}
                  cx="50%" cy="50%"
                  innerRadius={55} outerRadius={85}
                  paddingAngle={3} dataKey="value"
                  stroke="none"
                >
                  {chartData.resolutionBreakdown.map((entry, i) => (
                    <Cell key={i} fill={entry.fill} />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {chartData.resolutionBreakdown.map((item) => (
                <div key={item.name} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <div style={{ width: 10, height: 10, borderRadius: 3, background: item.fill }} />
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', flex: 1, minWidth: 110 }}>{item.name}</span>
                  <span style={{ fontSize: '0.9rem', fontWeight: 700 }}>{item.value}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Row */}
      <div className="grid-2">
        {/* Live Feed */}
        <div className="card" style={{ maxHeight: 360, overflow: 'hidden' }}>
          <div className="card-header">
            <span className="card-title">Live Ticket Feed</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#10b981', animation: 'pulse-glow 2s infinite' }} />
              <span style={{ fontSize: '0.7rem', color: '#34d399' }}>Real-time</span>
            </div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, overflowY: 'auto', maxHeight: 280 }}>
            {tickets.slice(0, 8).map((ticket, i) => (
              <div key={ticket.id}
                onClick={() => navigate(`/ticket/${ticket.id}`)}
                style={{
                  display: 'flex', alignItems: 'center', gap: 12,
                  padding: '10px 12px', borderRadius: 10,
                  background: 'rgba(15,23,42,0.4)',
                  border: '1px solid var(--border-primary)',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  animation: `fadeInUp 0.3s ease forwards`,
                  animationDelay: `${i * 0.05}s`,
                  opacity: 0,
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.borderColor = 'var(--border-hover)'
                  e.currentTarget.style.background = 'rgba(6,182,212,0.04)'
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.borderColor = 'var(--border-primary)'
                  e.currentTarget.style.background = 'rgba(15,23,42,0.4)'
                }}
              >
                <ChannelIcon channel={ticket.channel} size={14} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: '0.8rem', fontWeight: 500, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {ticket.subject}
                  </div>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 2 }}>
                    {ticket.customer.name} • {ticket.id}
                  </div>
                </div>
                <StatusBadge value={ticket.status} />
              </div>
            ))}
          </div>
        </div>

        {/* Fraud Alerts */}
        <div className="card" style={{ maxHeight: 360 }}>
          <div className="card-header">
            <span className="card-title">Fraud Alerts</span>
            <span className="badge badge-danger">
              <AlertTriangle size={10} /> {tickets.filter(t => t.trustScore < 50).length} flagged
            </span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {tickets.filter(t => t.trustScore < 70).map(ticket => (
              <div key={ticket.id}
                onClick={() => navigate(`/ticket/${ticket.id}`)}
                style={{
                  padding: '12px 14px', borderRadius: 10, cursor: 'pointer',
                  background: ticket.trustScore < 30 ? 'rgba(239,68,68,0.06)' : 'rgba(245,158,11,0.04)',
                  border: `1px solid ${ticket.trustScore < 30 ? 'rgba(239,68,68,0.15)' : 'rgba(245,158,11,0.12)'}`,
                  transition: 'all 0.2s ease',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
                  <span style={{ fontSize: '0.8rem', fontWeight: 600 }}>{ticket.id} — {ticket.customer.name}</span>
                  <span style={{
                    fontSize: '0.85rem', fontWeight: 800,
                    color: ticket.trustScore < 30 ? '#ef4444' : '#f59e0b'
                  }}>
                    {ticket.trustScore}
                  </span>
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: 6 }}>{ticket.subject}</div>
                {ticket.aiAnalysis.fraud.riskFactors.length > 0 && (
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                    {ticket.aiAnalysis.fraud.riskFactors.map((f, i) => (
                      <span key={i} style={{
                        fontSize: '0.62rem', padding: '2px 8px', borderRadius: 20,
                        background: 'rgba(239,68,68,0.08)', color: '#f87171',
                        border: '1px solid rgba(239,68,68,0.12)',
                      }}>{f}</span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
