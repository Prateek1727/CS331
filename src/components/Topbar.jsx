import { Search, Bell, Wifi } from 'lucide-react'

export default function Topbar() {
  return (
    <header style={{
      height: 64, minHeight: 64,
      background: 'rgba(12, 17, 32, 0.8)',
      backdropFilter: 'blur(20px)',
      borderBottom: '1px solid var(--border-primary)',
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '0 32px',
    }}>
      {/* Search */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 8,
        background: 'rgba(15, 23, 42, 0.6)',
        border: '1px solid var(--border-primary)',
        borderRadius: 10, padding: '8px 16px', width: 360,
      }}>
        <Search size={16} color="var(--text-muted)" />
        <input
          type="text"
          placeholder="Search tickets, customers, policies..."
          style={{
            background: 'none', border: 'none', outline: 'none',
            color: 'var(--text-primary)', fontSize: '0.825rem',
            width: '100%', fontFamily: 'Inter',
          }}
        />
        <kbd style={{
          fontSize: '0.6rem', color: 'var(--text-muted)',
          background: 'rgba(148,163,184,0.08)',
          padding: '2px 6px', borderRadius: 4,
          border: '1px solid var(--border-primary)',
        }}>⌘K</kbd>
      </div>

      {/* Right Side */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
        {/* System Status */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: '6px 14px', borderRadius: 20,
          background: 'rgba(16, 185, 129, 0.08)',
          border: '1px solid rgba(16, 185, 129, 0.15)',
        }}>
          <div style={{
            width: 7, height: 7, borderRadius: '50%', background: '#10b981',
            boxShadow: '0 0 8px rgba(16,185,129,0.5)',
            animation: 'pulse-glow 2s infinite',
          }} />
          <span style={{ fontSize: '0.72rem', color: '#34d399', fontWeight: 500 }}>
            All Systems Operational
          </span>
        </div>

        {/* Notifications */}
        <button style={{
          position: 'relative',
          width: 38, height: 38, borderRadius: 10,
          background: 'rgba(15, 23, 42, 0.6)',
          border: '1px solid var(--border-primary)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <Bell size={18} color="var(--text-secondary)" />
          <div style={{
            position: 'absolute', top: -2, right: -2,
            width: 16, height: 16, borderRadius: '50%',
            background: '#ef4444', fontSize: '0.55rem',
            fontWeight: 700, display: 'flex', alignItems: 'center', justifyContent: 'center',
            color: 'white', border: '2px solid var(--bg-secondary)',
          }}>3</div>
        </button>

        {/* Live Indicator */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 6,
          padding: '6px 12px', borderRadius: 8,
          background: 'rgba(6,182,212,0.06)',
          border: '1px solid rgba(6,182,212,0.12)',
        }}>
          <Wifi size={14} color="#06b6d4" />
          <span style={{ fontSize: '0.7rem', color: '#06b6d4', fontWeight: 600 }}>LIVE</span>
        </div>
      </div>
    </header>
  )
}
