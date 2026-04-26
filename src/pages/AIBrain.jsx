import { useState } from 'react'
import { Brain, Eye, BookOpen, Shield, Activity, Loader2, CheckCircle2, AlertTriangle } from 'lucide-react'
import TrustScoreGauge from '../components/TrustScoreGauge'
import { useData } from '../context/DataContext'
import { SENTIMENTS } from '../data/constants'

const ModelCard = ({ icon: Icon, title, color, children, delay = 0 }) => (
  <div className="card" style={{ animation: `fadeInUp 0.5s ease forwards`, animationDelay: `${delay}s`, opacity: 0 }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
      <div style={{
        width: 36, height: 36, borderRadius: 10,
        background: `${color}15`, border: `1px solid ${color}30`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        <Icon size={18} color={color} />
      </div>
      <div>
        <div style={{ fontSize: '0.9rem', fontWeight: 700 }}>{title}</div>
      </div>
    </div>
    {children}
  </div>
)

const ConfidenceBar = ({ label, value, color }) => (
  <div style={{ marginBottom: 10 }}>
    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: 4 }}>
      <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
      <span style={{ fontWeight: 600, color }}>{(value * 100).toFixed(0)}%</span>
    </div>
    <div style={{ height: 6, borderRadius: 3, background: 'rgba(148,163,184,0.1)' }}>
      <div style={{
        height: '100%', borderRadius: 3, background: color,
        width: `${value * 100}%`, transition: 'width 1s ease',
      }} />
    </div>
  </div>
)

export default function AIBrain() {
  const { tickets } = useData()
  const [selectedTicket, setSelectedTicket] = useState(tickets.length > 0 ? tickets[0].id : '')
  const ticket = tickets.find(t => t.id === selectedTicket)
  const analysis = ticket?.aiAnalysis

  return (
    <div className="animate-fade-in">
      <div className="page-header">
        <h1>AI Brain — Analysis Engine</h1>
        <p>Layer 2 — Four specialist models running in parallel on every ticket</p>
      </div>

      {/* Ticket Selector */}
      <div style={{ marginBottom: 24, display: 'flex', alignItems: 'center', gap: 12 }}>
        <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Analyzing ticket:</span>
        <select
          value={selectedTicket} onChange={e => setSelectedTicket(e.target.value)}
          style={{
            background: 'var(--bg-card)', border: '1px solid var(--border-primary)',
            borderRadius: 8, padding: '8px 12px', color: 'var(--text-primary)',
            fontSize: '0.82rem', fontFamily: 'Inter', outline: 'none', minWidth: 320,
          }}
        >
          {tickets.map(t => (
            <option key={t.id} value={t.id}>{t.id} — {t.subject.substring(0, 50)}...</option>
          ))}
        </select>
      </div>

      {/* Processing Pipeline */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 12, marginBottom: 24,
        padding: '14px 20px', borderRadius: 12,
        background: 'rgba(6,182,212,0.04)', border: '1px solid rgba(6,182,212,0.1)',
      }}>
        <Activity size={18} color="#06b6d4" />
        <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>Processing Pipeline:</span>
        {['NLP', 'Vision', 'RAG', 'Fraud'].map((stage, i) => (
          <span key={stage} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            {i > 0 && <span style={{ color: 'var(--text-muted)', margin: '0 4px' }}>→</span>}
            <CheckCircle2 size={14} color="#10b981" />
            <span style={{ fontSize: '0.78rem', fontWeight: 500, color: '#34d399' }}>{stage}</span>
          </span>
        ))}
        <span style={{ marginLeft: 'auto', fontSize: '0.72rem', color: 'var(--text-muted)' }}>
          Processed in 16s
        </span>
      </div>

      {/* 4 Model Cards */}
      <div className="grid-2">
        {/* NLP Engine */}
        <ModelCard icon={Brain} title="NLP Engine" color="#3b82f6" delay={0.1}>
          {analysis && (
            <>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 14 }}>
                <span className="badge badge-info">Intent: {analysis.nlp.intent.replace(/_/g, ' ')}</span>
                <span style={{ fontSize: '0.78rem', color: SENTIMENTS[analysis.nlp.sentiment]?.color }}>
                  {SENTIMENTS[analysis.nlp.sentiment]?.icon} {analysis.nlp.sentiment}
                </span>
                <span className="badge badge-neutral">Lang: {analysis.nlp.language.toUpperCase()}</span>
              </div>
              <ConfidenceBar label="Intent Confidence" value={analysis.nlp.confidence} color="#3b82f6" />
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 8 }}>
                <span style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>Entities: </span>
                {analysis.nlp.entities.join(', ')}
              </div>
            </>
          )}
        </ModelCard>

        {/* Vision Model */}
        <ModelCard icon={Eye} title="Vision Model — Forensic Analysis" color="#8b5cf6" delay={0.2}>
          {analysis?.vision ? (
            <>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 14 }}>
                <div style={{
                  padding: '6px 14px', borderRadius: 8,
                  background: analysis.vision.tamperingScore > 0.5 ? 'rgba(239,68,68,0.1)' : 'rgba(16,185,129,0.1)',
                  border: `1px solid ${analysis.vision.tamperingScore > 0.5 ? 'rgba(239,68,68,0.2)' : 'rgba(16,185,129,0.2)'}`,
                  fontSize: '0.82rem', fontWeight: 600,
                  color: analysis.vision.tamperingScore > 0.5 ? '#f87171' : '#34d399',
                }}>
                  {analysis.vision.tamperingScore > 0.5 ? <AlertTriangle size={14} style={{ marginRight: 6, verticalAlign: 'middle' }} /> : <CheckCircle2 size={14} style={{ marginRight: 6, verticalAlign: 'middle' }} />}
                  {analysis.vision.verdict}
                </div>
              </div>
              <ConfidenceBar label="Tampering Probability" value={analysis.vision.tamperingScore} color={analysis.vision.tamperingScore > 0.5 ? '#ef4444' : '#10b981'} />
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 10 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem' }}>
                  <span style={{ color: 'var(--text-muted)' }}>ELA Anomaly</span>
                  <span style={{ color: analysis.vision.elaAnomaly ? '#ef4444' : '#10b981', fontWeight: 600 }}>
                    {analysis.vision.elaAnomaly ? 'Detected' : 'None'}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Metadata Consistent</span>
                  <span style={{ color: analysis.vision.metadataConsistent ? '#10b981' : '#ef4444', fontWeight: 600 }}>
                    {analysis.vision.metadataConsistent ? 'Yes' : 'No'}
                  </span>
                </div>
                {analysis.vision.aiVision && (
                  <>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', marginTop: 6, paddingTop: 6, borderTop: '1px solid var(--border-primary)' }}>
                      <span style={{ color: 'var(--text-muted)' }}>AI Fraud Risk</span>
                      <span style={{ 
                        color: analysis.vision.aiVision.fraudRisk === 'high' ? '#ef4444' : analysis.vision.aiVision.fraudRisk === 'medium' ? '#f59e0b' : '#10b981', 
                        fontWeight: 600,
                        textTransform: 'uppercase',
                        fontSize: '0.72rem'
                      }}>
                        {analysis.vision.aiVision.fraudRisk}
                      </span>
                    </div>
                    {analysis.vision.aiVision.description && (
                      <div style={{ 
                        marginTop: 8, 
                        padding: '8px 10px', 
                        background: 'rgba(139,92,246,0.05)', 
                        borderRadius: 6,
                        fontSize: '0.72rem',
                        color: 'var(--text-secondary)',
                        lineHeight: 1.4
                      }}>
                        <strong>AI Analysis:</strong> {analysis.vision.aiVision.description}
                      </div>
                    )}
                  </>
                )}
              </div>
            </>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: 16, color: 'var(--text-muted)', fontSize: '0.82rem' }}>
              <span>No image attached — Vision model skipped</span>
            </div>
          )}
        </ModelCard>

        {/* Knowledge RAG */}
        <ModelCard icon={BookOpen} title="Knowledge RAG" color="#06b6d4" delay={0.3}>
          {analysis && (
            <>
              <ConfidenceBar label="Policy Match Confidence" value={analysis.rag.confidence} color="#06b6d4" />
              <div style={{ marginTop: 12 }}>
                <div style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Matched Policies
                </div>
                {analysis.rag.matchedPolicies.map((policy, i) => (
                  <div key={i} style={{
                    display: 'flex', alignItems: 'center', gap: 8,
                    padding: '8px 12px', borderRadius: 8, marginBottom: 6,
                    background: 'rgba(6,182,212,0.04)', border: '1px solid rgba(6,182,212,0.08)',
                  }}>
                    <BookOpen size={12} color="#06b6d4" />
                    <span style={{ fontSize: '0.78rem' }}>{policy}</span>
                  </div>
                ))}
              </div>
            </>
          )}
        </ModelCard>

        {/* Fraud Detector */}
        <ModelCard icon={Shield} title="Fraud Detector" color={ticket?.trustScore < 50 ? '#ef4444' : '#10b981'} delay={0.4}>
          {analysis && (
            <div style={{ display: 'flex', gap: 20 }}>
              <TrustScoreGauge score={ticket.trustScore} size={110} />
              <div style={{ flex: 1 }}>
                <div style={{
                  padding: '6px 12px', borderRadius: 8, marginBottom: 10,
                  background: analysis.fraud.verdict === 'High Risk' ? 'rgba(239,68,68,0.1)' : analysis.fraud.verdict === 'Medium Risk' ? 'rgba(245,158,11,0.1)' : 'rgba(16,185,129,0.1)',
                  border: `1px solid ${analysis.fraud.verdict === 'High Risk' ? 'rgba(239,68,68,0.2)' : analysis.fraud.verdict === 'Medium Risk' ? 'rgba(245,158,11,0.2)' : 'rgba(16,185,129,0.2)'}`,
                  fontSize: '0.78rem', fontWeight: 600,
                  color: analysis.fraud.verdict === 'High Risk' ? '#f87171' : analysis.fraud.verdict === 'Medium Risk' ? '#fbbf24' : '#34d399',
                  textAlign: 'center',
                }}>
                  {analysis.fraud.verdict}
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: 4 }}>
                  Anomalies: <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{analysis.fraud.anomalies}</span>
                </div>
                {analysis.fraud.riskFactors.length > 0 && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginTop: 8 }}>
                    {analysis.fraud.riskFactors.map((f, i) => (
                      <span key={i} style={{
                        fontSize: '0.68rem', padding: '3px 8px', borderRadius: 6,
                        background: 'rgba(239,68,68,0.06)', color: '#f87171',
                        border: '1px solid rgba(239,68,68,0.1)',
                      }}>⚠ {f}</span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </ModelCard>
      </div>
    </div>
  )
}
