import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft, Brain, Eye, BookOpen, Shield, Clock, CheckCircle2, AlertTriangle, Zap, Send, UserPlus, FileText, MessageSquare, Image } from 'lucide-react'
import TrustScoreGauge from '../components/TrustScoreGauge'
import ChannelIcon from '../components/ChannelIcon'
import StatusBadge from '../components/StatusBadge'
import { useData } from '../context/DataContext'
import { ticketService } from '../services/apiService'
import { SENTIMENTS } from '../data/constants'
import { useState } from 'react'

const timelineIcons = {
  created: Clock,
  processing: Brain,
  ai: Zap,
  decision: CheckCircle2,
  action: Send,
  resolved: CheckCircle2,
  escalated: AlertTriangle,
}
const timelineColors = {
  created: '#3b82f6',
  processing: '#8b5cf6',
  ai: '#06b6d4',
  decision: '#f59e0b',
  action: '#10b981',
  resolved: '#10b981',
  escalated: '#ef4444',
}

export default function TicketDetail() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [activeTab, setActiveTab] = useState('analysis')
  const [loading, setLoading] = useState(false)
  const { tickets } = useData()
  const ticket = tickets.find(t => t.id === id)

  const handleApproveAndSend = async () => {
    setLoading(true)
    try {
      const result = await ticketService.approveTicket(ticket.id)
      if (result.success) {
        alert(`✅ SUCCESS!\n\nTicket ${ticket.id} approved and customer notified!\n\n` +
              `Status: ${result.status}\n` +
              `Notifications sent:\n` +
              `- Email: ${result.notifications?.email_sent ? '✅' : '❌'}\n` +
              `- SMS: ${result.notifications?.sms_sent ? '✅' : '❌'}\n` +
              `- Webhook: ${result.notifications?.webhook_sent ? '✅' : '❌'}\n` +
              `- Slack: ${result.notifications?.slack_sent ? '✅' : '❌'}`)
        window.location.reload()
      } else {
        alert(`❌ Error: ${result.error}`)
      }
    } catch (error) {
      alert(`❌ Failed: ${error.message}`)
    } finally {
      setLoading(false)
    }
  }

  if (!ticket) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '60vh', gap: 16 }}>
        <h2 style={{ color: 'var(--text-muted)' }}>Ticket not found</h2>
        <button className="btn btn-primary" onClick={() => navigate('/tickets')}>
          <ArrowLeft size={16} /> Back to Tickets
        </button>
      </div>
    )
  }

  const sentConfig = SENTIMENTS[ticket.sentiment]

  return (
    <div className="animate-fade-in">
      {/* Back + Header */}
      <button onClick={() => navigate(-1)} style={{
        display: 'flex', alignItems: 'center', gap: 6,
        color: 'var(--text-muted)', fontSize: '0.82rem', marginBottom: 16,
        padding: '6px 0', transition: 'color 0.2s',
      }}
        onMouseEnter={e => e.currentTarget.style.color = 'var(--accent-primary)'}
        onMouseLeave={e => e.currentTarget.style.color = 'var(--text-muted)'}
      >
        <ArrowLeft size={16} /> Back
      </button>

      {/* Ticket Header Card */}
      <div className="card" style={{ marginBottom: 24, padding: '20px 24px' }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start' }}>
            <ChannelIcon channel={ticket.channel} size={20} />
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 4 }}>
                <h2 style={{ fontSize: '1.2rem', fontWeight: 700, margin: 0 }}>{ticket.id}</h2>
                <StatusBadge value={ticket.status} />
                <StatusBadge type="priority" value={ticket.priority} />
              </div>
              <p style={{ fontSize: '0.92rem', color: 'var(--text-secondary)', margin: '4px 0 10px' }}>{ticket.subject}</p>
              <div style={{ display: 'flex', alignItems: 'center', gap: 16, fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                <span><strong style={{ color: 'var(--text-primary)' }}>{ticket.customer.name}</strong> • {ticket.customer.email}</span>
                <span className={`badge ${ticket.customer.tier === 'VIP' ? 'badge-warning' : ticket.customer.tier === 'Premium' ? 'badge-purple' : 'badge-neutral'}`}>
                  {ticket.customer.tier}
                </span>
                <span style={{ color: sentConfig?.color }}>{sentConfig?.icon} {sentConfig?.label}</span>
              </div>
            </div>
          </div>
          <TrustScoreGauge score={ticket.trustScore} size={100} />
        </div>

        {/* Customer Message */}
        <div style={{
          marginTop: 18, padding: '14px 18px', borderRadius: 10,
          background: 'rgba(15,23,42,0.5)', border: '1px solid var(--border-primary)',
          fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.6,
        }}>
          <MessageSquare size={14} style={{ marginRight: 6, verticalAlign: 'middle', color: 'var(--text-muted)' }} />
          "{ticket.message}"
        </div>
      </div>

      {/* Tabs */}
      <div className="tabs" style={{ marginBottom: 24 }}>
        {['analysis', 'timeline', 'actions'].map(tab => (
          <button key={tab} className={`tab ${activeTab === tab ? 'active' : ''}`} onClick={() => setActiveTab(tab)}>
            {tab === 'analysis' ? 'AI Analysis' : tab === 'timeline' ? 'Timeline' : 'Actions'}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'analysis' && (
        <div className="grid-2" style={{ animationName: 'fadeInUp', animationDuration: '0.4s' }}>
          {/* NLP */}
          <div className="card">
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
              <Brain size={18} color="#3b82f6" />
              <span style={{ fontWeight: 700 }}>NLP Engine</span>
            </div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 12 }}>
              <span className="badge badge-info">Intent: {ticket.aiAnalysis.nlp.intent.replace(/_/g, ' ')}</span>
              <span style={{ color: sentConfig?.color, fontSize: '0.78rem' }}>{sentConfig?.icon} {ticket.sentiment}</span>
            </div>
            <div style={{ marginBottom: 8 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: 4 }}>
                <span style={{ color: 'var(--text-secondary)' }}>Confidence</span>
                <span style={{ fontWeight: 600, color: '#3b82f6' }}>{(ticket.aiAnalysis.nlp.confidence * 100).toFixed(0)}%</span>
              </div>
              <div style={{ height: 6, borderRadius: 3, background: 'rgba(148,163,184,0.1)' }}>
                <div style={{ height: '100%', borderRadius: 3, background: '#3b82f6', width: `${ticket.aiAnalysis.nlp.confidence * 100}%`, transition: 'width 1s ease' }} />
              </div>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              <strong style={{ color: 'var(--text-secondary)' }}>Entities:</strong> {ticket.aiAnalysis.nlp.entities.join(', ')}
            </div>
          </div>

          {/* Vision */}
          <div className="card">
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
              <Eye size={18} color="#8b5cf6" />
              <span style={{ fontWeight: 700 }}>Vision Model</span>
            </div>
            {ticket.aiAnalysis.vision ? (
              <>
                <div style={{
                  display: 'inline-flex', alignItems: 'center', gap: 6,
                  padding: '6px 12px', borderRadius: 8, marginBottom: 12,
                  background: ticket.aiAnalysis.vision.tamperingScore > 0.5 ? 'rgba(239,68,68,0.1)' : 'rgba(16,185,129,0.1)',
                  color: ticket.aiAnalysis.vision.tamperingScore > 0.5 ? '#f87171' : '#34d399',
                  fontSize: '0.8rem', fontWeight: 600,
                }}>
                  {ticket.aiAnalysis.vision.tamperingScore > 0.5 ? <AlertTriangle size={14} /> : <CheckCircle2 size={14} />}
                  {ticket.aiAnalysis.vision.verdict}
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: '0.78rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Tampering Score</span>
                    <span style={{ fontWeight: 600, color: ticket.aiAnalysis.vision.tamperingScore > 0.5 ? '#ef4444' : '#10b981' }}>
                      {(ticket.aiAnalysis.vision.tamperingScore * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--text-muted)' }}>ELA Anomaly</span>
                    <span style={{ color: ticket.aiAnalysis.vision.elaAnomaly ? '#ef4444' : '#10b981', fontWeight: 600 }}>
                      {ticket.aiAnalysis.vision.elaAnomaly ? 'Detected' : 'None'}
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Metadata</span>
                    <span style={{ color: ticket.aiAnalysis.vision.metadataConsistent ? '#10b981' : '#ef4444', fontWeight: 600 }}>
                      {ticket.aiAnalysis.vision.metadataConsistent ? 'Consistent' : 'Stripped/Modified'}
                    </span>
                  </div>
                </div>
              </>
            ) : (
              <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>No image attached — skipped</p>
            )}
          </div>

          {/* RAG */}
          <div className="card">
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
              <BookOpen size={18} color="#06b6d4" />
              <span style={{ fontWeight: 700 }}>Knowledge RAG</span>
            </div>
            <div style={{ marginBottom: 12 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: 4 }}>
                <span style={{ color: 'var(--text-secondary)' }}>Match Confidence</span>
                <span style={{ fontWeight: 600, color: '#06b6d4' }}>{(ticket.aiAnalysis.rag.confidence * 100).toFixed(0)}%</span>
              </div>
              <div style={{ height: 6, borderRadius: 3, background: 'rgba(148,163,184,0.1)' }}>
                <div style={{ height: '100%', borderRadius: 3, background: '#06b6d4', width: `${ticket.aiAnalysis.rag.confidence * 100}%` }} />
              </div>
            </div>
            {ticket.aiAnalysis.rag.matchedPolicies.map((p, i) => (
              <div key={i} style={{
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '8px 10px', borderRadius: 8, marginBottom: 4,
                background: 'rgba(6,182,212,0.04)', border: '1px solid rgba(6,182,212,0.08)',
                fontSize: '0.75rem',
              }}>
                <FileText size={12} color="#06b6d4" /> {p}
              </div>
            ))}
          </div>

          {/* Fraud */}
          <div className="card">
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
              <Shield size={18} color={ticket.trustScore < 50 ? '#ef4444' : '#10b981'} />
              <span style={{ fontWeight: 700 }}>Fraud Detector</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
              <TrustScoreGauge score={ticket.trustScore} size={90} />
              <div>
                <div style={{
                  padding: '4px 10px', borderRadius: 6, marginBottom: 8,
                  background: ticket.aiAnalysis.fraud.verdict === 'High Risk' ? 'rgba(239,68,68,0.1)' : ticket.aiAnalysis.fraud.verdict === 'Medium Risk' ? 'rgba(245,158,11,0.1)' : 'rgba(16,185,129,0.1)',
                  color: ticket.aiAnalysis.fraud.verdict === 'High Risk' ? '#f87171' : ticket.aiAnalysis.fraud.verdict === 'Medium Risk' ? '#fbbf24' : '#34d399',
                  fontSize: '0.75rem', fontWeight: 600,
                }}>
                  {ticket.aiAnalysis.fraud.verdict}
                </div>
                {ticket.aiAnalysis.fraud.riskFactors.map((f, i) => (
                  <div key={i} style={{ fontSize: '0.72rem', color: '#f87171', marginBottom: 2 }}>⚠ {f}</div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'timeline' && (
        <div className="card" style={{ animationName: 'fadeInUp', animationDuration: '0.4s' }}>
          <div style={{ position: 'relative', paddingLeft: 32 }}>
            {/* Vertical line */}
            <div style={{
              position: 'absolute', left: 11, top: 8, bottom: 8, width: 2,
              background: 'linear-gradient(180deg, #06b6d4, #8b5cf6, #10b981)',
              borderRadius: 2, opacity: 0.3,
            }} />

            {ticket.timeline.map((event, i) => {
              const Icon = timelineIcons[event.type] || Clock
              const color = timelineColors[event.type] || '#64748b'
              return (
                <div key={i} style={{
                  position: 'relative', marginBottom: 20, paddingLeft: 20,
                  animation: `fadeInUp 0.4s ease forwards`,
                  animationDelay: `${i * 0.1}s`,
                  opacity: 0,
                }}>
                  {/* Dot */}
                  <div style={{
                    position: 'absolute', left: -26, top: 3,
                    width: 24, height: 24, borderRadius: '50%',
                    background: `${color}15`, border: `2px solid ${color}`,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                  }}>
                    <Icon size={11} color={color} />
                  </div>

                  <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
                    <div>
                      <div style={{ fontSize: '0.85rem', fontWeight: 500 }}>{event.event}</div>
                    </div>
                    <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', whiteSpace: 'nowrap', marginLeft: 16 }}>
                      {event.time}
                    </span>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {activeTab === 'actions' && (
        <div style={{ animationName: 'fadeInUp', animationDuration: '0.4s' }}>
          {/* Decision Summary */}
          <div className="card" style={{ marginBottom: 16 }}>
            <div className="card-header">
              <span className="card-title">AI Decision</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
              <span className={`badge ${ticket.decision.action === 'auto_resolve' ? 'badge-success' : ticket.decision.action === 'escalate' ? 'badge-danger' : 'badge-warning'}`} style={{ fontSize: '0.82rem', padding: '6px 14px' }}>
                {ticket.decision.action.replace(/_/g, ' ').toUpperCase()}
              </span>
              <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                Confidence: <strong style={{ color: ticket.decision.confidence > 0.85 ? '#10b981' : '#f59e0b' }}>
                  {(ticket.decision.confidence * 100).toFixed(0)}%
                </strong>
              </span>
              <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>Rule: {ticket.decision.rule}</span>
            </div>
            {ticket.decision.refundAmount && (
              <div style={{
                marginTop: 12, padding: '8px 14px', borderRadius: 8,
                background: 'rgba(16,185,129,0.06)', border: '1px solid rgba(16,185,129,0.12)',
                display: 'inline-flex', alignItems: 'center', gap: 6,
                fontSize: '0.82rem', color: '#34d399', fontWeight: 600,
              }}>
                Refund Issued: ${ticket.decision.refundAmount}
              </div>
            )}
          </div>

          {/* Action Buttons */}
          {ticket.status !== 'resolved' && ticket.status !== 'rejected' && (
            <div style={{ display: 'flex', gap: 12 }}>
              <button 
                className="btn btn-primary" 
                onClick={handleApproveAndSend} 
                disabled={loading}
              >
                <Send size={14} /> {loading ? 'Processing...' : 'Approve AI Draft'}
              </button>
              <button className="btn btn-ghost"><UserPlus size={14} /> Escalate to Agent</button>
              <button className="btn btn-ghost"><CheckCircle2 size={14} /> Resolve</button>
              <button className="btn btn-ghost"><MessageSquare size={14} /> Add Note</button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
