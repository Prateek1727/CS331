import { NavLink, useLocation } from 'react-router-dom'
import {
  LayoutDashboard, Inbox, Brain, GitBranch, Zap, BarChart3,
  ChevronLeft, ChevronRight, Sparkles, UserCircle
} from 'lucide-react'
import { useState } from 'react'

const navItems = [
  { path: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { path: '/tickets', icon: Inbox, label: 'Tickets' },
  { path: '/ai-brain', icon: Brain, label: 'AI Brain' },
  { path: '/decision-engine', icon: GitBranch, label: 'Decision Engine' },
  { path: '/actions', icon: Zap, label: 'Actions' },
  { path: '/intelligence', icon: BarChart3, label: 'Intelligence' },
  { path: '/customer-portal', icon: UserCircle, label: 'Customer Portal' },
]

export default function Sidebar() {
  const [collapsed, setCollapsed] = useState(false)
  const location = useLocation()

  return (
    <aside style={{
      width: collapsed ? 72 : 260,
      minWidth: collapsed ? 72 : 260,
      height: '100vh',
      background: 'linear-gradient(180deg, #0c1120 0%, #060a13 100%)',
      borderRight: '1px solid var(--border-primary)',
      display: 'flex',
      flexDirection: 'column',
      transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
      position: 'relative',
      zIndex: 50,
    }}>
      {/* Logo */}
      <div style={{
        padding: collapsed ? '20px 12px' : '20px 24px',
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        borderBottom: '1px solid var(--border-primary)',
        minHeight: 64,
      }}>
        <div style={{
          width: 36, height: 36, borderRadius: 10,
          background: 'var(--gradient-primary)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          boxShadow: '0 4px 16px rgba(6,182,212,0.3)',
          flexShrink: 0,
        }}>
          <Sparkles size={20} color="white" />
        </div>
        {!collapsed && (
          <div style={{ overflow: 'hidden' }}>
            <div style={{ fontSize: '1.1rem', fontWeight: 800, letterSpacing: '-0.02em', background: 'var(--gradient-primary)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
              NeuraDesk
            </div>
            <div style={{ fontSize: '0.62rem', color: 'var(--text-muted)', fontWeight: 500, letterSpacing: '0.08em', textTransform: 'uppercase' }}>
              AI Support Platform
            </div>
          </div>
        )}
      </div>

      {/* Nav Items */}
      <nav style={{ flex: 1, padding: '16px 12px', display: 'flex', flexDirection: 'column', gap: 4 }}>
        {navItems.map(({ path, icon: Icon, label }) => {
          const isActive = location.pathname === path
          return (
            <NavLink key={path} to={path} style={{
              display: 'flex', alignItems: 'center', gap: 12,
              padding: collapsed ? '12px' : '10px 16px',
              borderRadius: 10,
              color: isActive ? '#fff' : 'var(--text-muted)',
              background: isActive ? 'rgba(6,182,212,0.1)' : 'transparent',
              border: isActive ? '1px solid rgba(6,182,212,0.2)' : '1px solid transparent',
              transition: 'all 0.2s ease',
              textDecoration: 'none',
              position: 'relative',
              justifyContent: collapsed ? 'center' : 'flex-start',
            }}>
              {isActive && (
                <div style={{
                  position: 'absolute', left: -12, top: '50%', transform: 'translateY(-50%)',
                  width: 3, height: 24, borderRadius: 4,
                  background: 'var(--gradient-primary)',
                }} />
              )}
              <Icon size={20} style={{ flexShrink: 0, color: isActive ? '#06b6d4' : undefined }} />
              {!collapsed && (
                <span style={{ fontSize: '0.85rem', fontWeight: isActive ? 600 : 400 }}>{label}</span>
              )}
            </NavLink>
          )
        })}
      </nav>

      {/* Collapse Toggle */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        style={{
          position: 'absolute', right: -14, top: 80,
          width: 28, height: 28, borderRadius: '50%',
          background: 'var(--bg-tertiary)', border: '1px solid var(--border-primary)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: 'var(--text-muted)', zIndex: 60,
          transition: 'all 0.2s ease',
        }}
      >
        {collapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
      </button>

      {/* Bottom Profile */}
      <div style={{
        padding: collapsed ? '16px 12px' : '16px 20px',
        borderTop: '1px solid var(--border-primary)',
        display: 'flex', alignItems: 'center', gap: 10,
        justifyContent: collapsed ? 'center' : 'flex-start',
      }}>
        <div style={{
          width: 34, height: 34, borderRadius: '50%',
          background: 'var(--gradient-accent)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: '0.75rem', fontWeight: 700, color: 'white', flexShrink: 0,
        }}>
          AD
        </div>
        {!collapsed && (
          <div>
            <div style={{ fontSize: '0.8rem', fontWeight: 600 }}>Admin User</div>
            <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>Platform Admin</div>
          </div>
        )}
      </div>
    </aside>
  )
}
