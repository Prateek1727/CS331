import { Mail, MessageCircle, Phone, Share2, Smartphone, Webhook } from 'lucide-react'
import { CHANNELS } from '../data/constants'

const iconMap = {
  email: Mail,
  chat: MessageCircle,
  voice: Phone,
  social: Share2,
  mobile: Smartphone,
  api: Webhook,
}

export default function ChannelIcon({ channel, size = 16 }) {
  const Icon = iconMap[channel] || Mail
  const config = CHANNELS[channel] || CHANNELS.email
  return (
    <div style={{
      display: 'inline-flex',
      alignItems: 'center',
      justifyContent: 'center',
      width: size + 14,
      height: size + 14,
      borderRadius: '8px',
      background: config.bg,
      border: `1px solid ${config.color}22`,
    }}>
      <Icon size={size} color={config.color} />
    </div>
  )
}
