import { STATUSES, PRIORITIES } from '../data/constants'

export default function StatusBadge({ type = 'status', value }) {
  const map = type === 'priority' ? PRIORITIES : STATUSES
  const config = map[value]
  if (!config) return <span className="badge badge-neutral">{value}</span>
  return <span className={`badge ${config.className}`}>{config.label}</span>
}
