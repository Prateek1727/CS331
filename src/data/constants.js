export const CHANNELS = {
  email: { label: 'Email', color: '#3b82f6', bg: 'rgba(59,130,246,0.12)' },
  chat: { label: 'Live Chat', color: '#10b981', bg: 'rgba(16,185,129,0.12)' },
  voice: { label: 'Voice', color: '#f59e0b', bg: 'rgba(245,158,11,0.12)' },
  social: { label: 'Social', color: '#ec4899', bg: 'rgba(236,72,153,0.12)' },
  mobile: { label: 'Mobile', color: '#8b5cf6', bg: 'rgba(139,92,246,0.12)' },
  api: { label: 'API', color: '#06b6d4', bg: 'rgba(6,182,212,0.12)' },
}

export const STATUSES = {
  open: { label: 'Open', className: 'badge-info' },
  in_progress: { label: 'In Progress', className: 'badge-warning' },
  auto_resolved: { label: 'Auto-Resolved', className: 'badge-success' },
  escalated: { label: 'Escalated', className: 'badge-danger' },
  pending_review: { label: 'Pending Review', className: 'badge-purple' },
  closed: { label: 'Closed', className: 'badge-neutral' },
}

export const PRIORITIES = {
  critical: { label: 'Critical', className: 'badge-danger' },
  high: { label: 'High', className: 'badge-warning' },
  medium: { label: 'Medium', className: 'badge-info' },
  low: { label: 'Low', className: 'badge-neutral' },
}

export const SENTIMENTS = {
  positive: { label: 'Positive', color: '#10b981', icon: '😊' },
  neutral: { label: 'Neutral', color: '#f59e0b', icon: '😐' },
  negative: { label: 'Negative', color: '#ef4444', icon: '😤' },
  angry: { label: 'Angry', color: '#dc2626', icon: '🔥' },
}
