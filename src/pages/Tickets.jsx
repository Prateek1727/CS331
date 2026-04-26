import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Search, Filter, Loader2 } from 'lucide-react'
import ChannelIcon from '../components/ChannelIcon'
import StatusBadge from '../components/StatusBadge'
import { ticketService } from '../services/apiService'
import { CHANNELS, SENTIMENTS } from '../data/constants'

export default function Tickets() {
  const [tickets, setTickets] = useState([])
  const [loading, setLoading] = useState(true)
  const [activeChannel, setActiveChannel] = useState('all')
  const [searchQuery, setSearchQuery] = useState('')
  const navigate = useNavigate()

  useEffect(() => {
    const fetchTickets = async () => {
      try {
        const data = await ticketService.getTickets()
        setTickets(data)
      } catch (err) {
        console.error(err)
      } finally {
        setLoading(false)
      }
    }
    fetchTickets()
    
    // Auto-refresh every 5 seconds
    const interval = setInterval(fetchTickets, 5000)
    return () => clearInterval(interval)
  }, [])

  const channels = ['all', ...Object.keys(CHANNELS)]
  const filtered = tickets.filter(t => {
    if (activeChannel !== 'all' && t.channel !== activeChannel) return false
    if (searchQuery && !t.subject.toLowerCase().includes(searchQuery.toLowerCase()) && !t.id.toLowerCase().includes(searchQuery.toLowerCase())) return false
    return true
  })

  return (
    <div className="animate-fade-in">
      <div className="page-header">
        <h1>Unified Ticket Inbox</h1>
        <p>Layer 1 — Channel Ingestion: All customer touchpoints normalized into one stream</p>
      </div>

      {/* Filters */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 24 }}>
        <div className="tabs">
          {channels.map(ch => (
            <button key={ch} className={`tab ${activeChannel === ch ? 'active' : ''}`} onClick={() => setActiveChannel(ch)}>
              {ch === 'all' ? 'All Channels' : CHANNELS[ch]?.label}
            </button>
          ))}
        </div>
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          background: 'rgba(15,23,42,0.6)', border: '1px solid var(--border-primary)',
          borderRadius: 8, padding: '7px 12px', marginLeft: 'auto',
        }}>
          <Search size={14} color="var(--text-muted)" />
          <input
            type="text" placeholder="Search tickets..."
            value={searchQuery} onChange={e => setSearchQuery(e.target.value)}
            style={{ background: 'none', border: 'none', outline: 'none', color: 'var(--text-primary)', fontSize: '0.8rem', width: 180, fontFamily: 'Inter' }}
          />
        </div>
      </div>

      {/* Table */}
      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <div className="table-container">
          {loading && tickets.length === 0 ? (
            <div style={{ padding: 40, textAlign: 'center', color: 'var(--text-muted)' }}>
              <Loader2 className="animate-spin" size={24} style={{ margin: '0 auto 16px', color: 'var(--accent-primary)' }} />
              Loading real-time tickets from API...
            </div>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>Ticket ID</th>
                  <th>Channel</th>
                  <th>Customer</th>
                  <th>Subject</th>
                  <th>Priority</th>
                  <th>Sentiment</th>
                  <th>Trust</th>
                  <th>Status</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((ticket, i) => {
                  const sentConfig = SENTIMENTS[ticket.sentiment]
                  return (
                    <tr key={ticket.id}
                      onClick={() => navigate(`/ticket/${ticket.id}`)}
                      style={{ cursor: 'pointer', animation: `fadeInUp 0.3s ease forwards`, animationDelay: `${i * 0.03}s`, opacity: 0 }}
                    >
                      <td style={{ fontWeight: 600, color: 'var(--accent-primary)', fontSize: '0.8rem' }}>{ticket.id}</td>
                      <td><ChannelIcon channel={ticket.channel} size={14} /></td>
                      <td>
                        <div style={{ fontSize: '0.82rem', fontWeight: 500 }}>{ticket.customer.name}</div>
                        <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>{ticket.customer.tier}</div>
                      </td>
                      <td style={{ maxWidth: 280, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', fontSize: '0.82rem' }}>
                        {ticket.subject}
                      </td>
                      <td><StatusBadge type="priority" value={ticket.priority} /></td>
                      <td>
                        <span style={{ fontSize: '0.78rem', color: sentConfig?.color }}>
                          {sentConfig?.icon} {sentConfig?.label}
                        </span>
                      </td>
                      <td>
                        <span style={{
                          fontWeight: 700, fontSize: '0.82rem',
                          color: ticket.trustScore >= 80 ? '#10b981' : ticket.trustScore >= 50 ? '#f59e0b' : '#ef4444'
                        }}>
                          {ticket.trustScore}
                        </span>
                      </td>
                      <td><StatusBadge value={ticket.status} /></td>
                      <td style={{ fontSize: '0.75rem', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                        {new Date(ticket.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </td>
                    </tr>
                  )
                })}
                {filtered.length === 0 && !loading && (
                   <tr><td colSpan={9} style={{ textAlign: 'center', padding: 32, color: 'var(--text-muted)' }}>No tickets match your filters.</td></tr>
                )}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {/* Summary */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: 16, padding: '0 4px' }}>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          Showing {filtered.length} of {tickets.length} tickets (Live API)
        </span>
        <div style={{ display: 'flex', gap: 12 }}>
          {Object.entries(CHANNELS).map(([key, ch]) => {
            const count = tickets.filter(t => t.channel === key).length
            return (
              <span key={key} style={{ fontSize: '0.7rem', color: ch.color, display: 'flex', alignItems: 'center', gap: 4 }}>
                <div style={{ width: 6, height: 6, borderRadius: 2, background: ch.color }} />
                {ch.label}: {count}
              </span>
            )
          })}
        </div>
      </div>
    </div>
  )
}
