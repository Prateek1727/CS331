import { CheckCircle2, FileEdit, UserPlus, Bell, Zap, DollarSign, Send, ExternalLink } from 'lucide-react'
import ChannelIcon from '../components/ChannelIcon'
import StatusBadge from '../components/StatusBadge'
import { useData } from '../context/DataContext'
import { ticketService } from '../services/apiService'
import { useState } from 'react'

const SectionCard = ({ icon: Icon, title, color, count, children }) => (
  <div className="card" style={{ flex: 1 }}>
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <div style={{
          width: 32, height: 32, borderRadius: 8,
          background: `${color}15`, border: `1px solid ${color}25`,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <Icon size={16} color={color} />
        </div>
        <span style={{ fontSize: '0.88rem', fontWeight: 700 }}>{title}</span>
      </div>
      <span style={{
        padding: '3px 10px', borderRadius: 20,
        background: `${color}12`, color, fontSize: '0.72rem', fontWeight: 700,
      }}>
        {count}
      </span>
    </div>
    {children}
  </div>
)

const TicketItem = ({ ticket, children }) => (
  <div style={{
    padding: '12px 14px', borderRadius: 10,
    background: 'rgba(15,23,42,0.4)', border: '1px solid var(--border-primary)',
    marginBottom: 8, transition: 'all 0.2s ease',
  }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
      <ChannelIcon channel={ticket.channel} size={12} />
      <span style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--accent-primary)' }}>{ticket.id}</span>
      <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{ticket.customer.name}</span>
    </div>
    <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginBottom: 8, lineHeight: 1.4 }}>
      {ticket.subject}
    </div>
    {children}
  </div>
)

export default function ActionLayer() {
  const { tickets } = useData()
  const [loading, setLoading] = useState({})
  const autoResolved = tickets.filter(t => t.status === 'auto_resolved')
  const drafts = tickets.filter(t => t.decision.action === 'draft_response')
  const escalated = tickets.filter(t => t.status === 'escalated')
  const inProgress = tickets.filter(t => t.status === 'in_progress')

  const handleApproveAndSend = async (ticketId) => {
    setLoading(prev => ({ ...prev, [ticketId]: true }))
    
    try {
      const result = await ticketService.approveTicket(ticketId)
      
      if (result.success) {
        alert(`✅ SUCCESS!\n\nTicket ${ticketId} approved and customer notified!\n\n` +
              `Status: ${result.status}\n` +
              `Notifications sent:\n` +
              `- Email: ${result.notifications.email_sent ? '✅' : '❌'}\n` +
              `- SMS: ${result.notifications.sms_sent ? '✅' : '❌'}\n` +
              `- Webhook: ${result.notifications.webhook_sent ? '✅' : '❌'}\n` +
              `- Slack: ${result.notifications.slack_sent ? '✅' : '❌'}`)
        
        // Refresh tickets
        window.location.reload()
      } else {
        alert(`❌ Error: ${result.error}`)
      }
    } catch (error) {
      alert(`❌ Failed to approve ticket: ${error.message}`)
    } finally {
      setLoading(prev => ({ ...prev, [ticketId]: false }))
    }
  }

  const handleEditDraft = async (ticketId) => {
    const newMessage = prompt('Enter new draft message:', 
      'Thank you for reaching out. We understand your concern and our team is looking into this. We\'ll have a resolution for you shortly.')
    
    if (!newMessage) return
    
    setLoading(prev => ({ ...prev, [ticketId]: true }))
    
    try {
      const result = await ticketService.updateDraft(ticketId, newMessage)
      
      if (result.success) {
        alert(`✅ Draft updated successfully for ticket ${ticketId}!`)
        window.location.reload()
      } else {
        alert(`❌ Error: ${result.error}`)
      }
    } catch (error) {
      alert(`❌ Failed to update draft: ${error.message}`)
    } finally {
      setLoading(prev => ({ ...prev, [ticketId]: false }))
    }
  }

  return (
    <div className="animate-fade-in">
      <div className="page-header">
        <h1>Action Layer</h1>
        <p>Layer 4 — Executing decisions: auto-resolve, draft, handoff, or notify</p>
      </div>

      {/* Stats Row */}
      <div style={{ display: 'flex', gap: 16, marginBottom: 24 }}>
        {[
          { label: 'Auto-Resolved Today', value: '247', icon: CheckCircle2, color: '#10b981' },
          { label: 'Drafts Pending', value: String(drafts.length), icon: FileEdit, color: '#f59e0b' },
          { label: 'Agent Handoffs', value: String(escalated.length), icon: UserPlus, color: '#8b5cf6' },
          { label: 'Notifications Sent', value: '892', icon: Bell, color: '#06b6d4' },
          { label: 'Refunds Issued', value: '$12,450', icon: DollarSign, color: '#ec4899' },
        ].map(({ label, value, icon: Icon, color }) => (
          <div key={label} style={{
            flex: 1, padding: '14px 16px', borderRadius: 12,
            background: `${color}06`, border: `1px solid ${color}12`,
            display: 'flex', alignItems: 'center', gap: 10,
          }}>
            <Icon size={18} color={color} />
            <div>
              <div style={{ fontSize: '1.1rem', fontWeight: 800, color }}>{value}</div>
              <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>{label}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="grid-2" style={{ marginBottom: 24 }}>
        {/* Auto-Resolve Feed */}
        <SectionCard icon={CheckCircle2} title="Auto-Resolved" color="#10b981" count={autoResolved.length}>
          {autoResolved.map(ticket => (
            <TicketItem key={ticket.id} ticket={ticket}>
              {ticket.decision.refundAmount && (
                <div style={{
                  display: 'inline-flex', alignItems: 'center', gap: 4,
                  padding: '4px 10px', borderRadius: 6,
                  background: 'rgba(16,185,129,0.1)', fontSize: '0.72rem',
                  color: '#34d399', fontWeight: 600,
                }}>
                  <DollarSign size={12} /> Refund: ${ticket.decision.refundAmount}
                </div>
              )}
              <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginTop: 4 }}>
                Resolved in {ticket.resolvedAt ? Math.round((new Date(ticket.resolvedAt) - new Date(ticket.createdAt)) / 1000) + 's' : '—'}
              </div>
            </TicketItem>
          ))}
        </SectionCard>

        {/* Draft Responses */}
        <SectionCard icon={FileEdit} title="Draft Responses" color="#f59e0b" count={drafts.length}>
          {drafts.map(ticket => (
            <TicketItem key={ticket.id} ticket={ticket}>
              <div style={{
                padding: '10px 12px', borderRadius: 8, marginBottom: 8,
                background: 'rgba(245,158,11,0.04)', border: '1px solid rgba(245,158,11,0.08)',
                fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.5, fontStyle: 'italic',
              }}>
                "Thank you for reaching out. We understand your concern about {ticket.aiAnalysis.nlp.entities[0]}. Our team is looking into this and we'll have a resolution for you shortly..."
              </div>
              <div style={{ display: 'flex', gap: 6 }}>
                <button 
                  className="btn btn-primary" 
                  style={{ fontSize: '0.72rem', padding: '5px 12px' }}
                  onClick={() => handleApproveAndSend(ticket.id)}
                  disabled={loading[ticket.id]}
                >
                  {loading[ticket.id] ? '⏳ Processing...' : <><Send size={12} /> Approve & Send</>}
                </button>
                <button 
                  className="btn btn-ghost" 
                  style={{ fontSize: '0.72rem', padding: '5px 12px' }}
                  onClick={() => handleEditDraft(ticket.id)}
                  disabled={loading[ticket.id]}
                >
                  Edit Draft
                </button>
              </div>
            </TicketItem>
          ))}
        </SectionCard>
      </div>

      <div className="grid-2">
        {/* Agent Handoff Queue */}
        <SectionCard icon={UserPlus} title="Agent Handoff Queue" color="#8b5cf6" count={escalated.length}>
          {escalated.map(ticket => (
            <TicketItem key={ticket.id} ticket={ticket}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                <StatusBadge type="priority" value={ticket.priority} />
                <span style={{
                  fontSize: '0.7rem', fontWeight: 600,
                  color: ticket.trustScore < 30 ? '#ef4444' : '#f59e0b',
                }}>
                  Trust: {ticket.trustScore}
                </span>
              </div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                Reason: {ticket.decision.rule}
              </div>
            </TicketItem>
          ))}
        </SectionCard>

        {/* Customer Notifications */}
        <SectionCard icon={Bell} title="Customer Notifications" color="#06b6d4" count={autoResolved.length}>
          {autoResolved.map(ticket => (
            <TicketItem key={ticket.id} ticket={ticket}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <ChannelIcon channel={ticket.channel} size={12} />
                <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                  Notified via {ticket.channel} at {ticket.resolvedAt ? new Date(ticket.resolvedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '—'}
                </span>
                <CheckCircle2 size={12} color="#10b981" style={{ marginLeft: 'auto' }} />
                <span style={{ fontSize: '0.68rem', color: '#34d399' }}>Delivered</span>
              </div>
            </TicketItem>
          ))}
        </SectionCard>
      </div>
    </div>
  )
}
