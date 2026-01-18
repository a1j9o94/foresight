import { useState, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  CheckCircle,
  Clock,
  AlertCircle,
  AlertTriangle,
  ExternalLink,
  ChevronDown,
  ChevronRight,
  Activity,
  FileText,
  BarChart3,
  RefreshCw,
  XCircle
} from 'lucide-react'

interface Metrics {
  lpips?: number
  ssim?: number
  spatial_iou?: number
  confidence?: string
  [key: string]: unknown
}

interface SubExperiment {
  id: string
  status: string
  finding: string
  metrics: Record<string, number>
  error?: string | null
  hasError?: boolean
}

interface Experiment {
  id: string
  name: string
  phase: number
  status: string
  recommendation?: string
  metrics: Metrics
  subExperiments: SubExperiment[]
  hasFindings: boolean
  hasPlan: boolean
  findings?: string
}

interface Agent {
  id: string
  experiment: string
  lastUpdate: string
  isActive: boolean
  outputFile: string
}

interface Gate {
  id: string
  name: string
  experiments: string[]
  unlocks: string
  status: string
  progress: string
  blockers?: string[]
  blockerReason?: 'errors' | 'pivot'
}

interface Status {
  generatedAt: string
  experiments: Experiment[]
  agents: Agent[]
  gates: Gate[]
  links: {
    wandb: string
    modal: string
    github: string
  }
}

function StatusBadge({ status, recommendation }: { status: string, recommendation?: string }) {
  if (status === 'completed' && recommendation === 'proceed') {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-sm bg-green-100 text-green-800">
        <CheckCircle size={14} /> Passed
      </span>
    )
  }
  if (status === 'completed_with_errors') {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-sm bg-orange-100 text-orange-800">
        <AlertTriangle size={14} /> Has Errors
      </span>
    )
  }
  if (status === 'completed') {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-sm bg-blue-100 text-blue-800">
        <CheckCircle size={14} /> Complete
      </span>
    )
  }
  if (status === 'failed') {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-sm bg-red-100 text-red-800">
        <XCircle size={14} /> Failed
      </span>
    )
  }
  if (status === 'running' || status === 'in_progress') {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-sm bg-yellow-100 text-yellow-800">
        <Clock size={14} /> Running
      </span>
    )
  }
  return (
    <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-sm bg-gray-100 text-gray-600">
      <AlertCircle size={14} /> Not Started
    </span>
  )
}

function MetricCard({ label, value, target, unit = '' }: { label: string, value?: number, target?: string, unit?: string }) {
  if (value === undefined) return null

  return (
    <div className="bg-white rounded-lg p-3 border border-gray-200">
      <div className="text-sm text-gray-500">{label}</div>
      <div className="text-2xl font-bold text-gray-900">{value.toFixed(3)}{unit}</div>
      {target && <div className="text-xs text-gray-400">Target: {target}</div>}
    </div>
  )
}

function ExperimentCard({ experiment, onViewFindings }: { experiment: Experiment, onViewFindings: (exp: Experiment) => void }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
      <div
        className="p-4 cursor-pointer hover:bg-gray-50 flex items-center justify-between"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-3">
          {expanded ? <ChevronDown size={20} /> : <ChevronRight size={20} />}
          <div>
            <h3 className="font-semibold text-gray-900">{experiment.name}</h3>
            <p className="text-sm text-gray-500">Phase {experiment.phase}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <StatusBadge status={experiment.status} recommendation={experiment.recommendation} />
          {experiment.hasFindings && (
            <button
              onClick={(e) => { e.stopPropagation(); onViewFindings(experiment); }}
              className="p-2 text-blue-600 hover:bg-blue-50 rounded-lg"
              title="View Findings"
            >
              <FileText size={18} />
            </button>
          )}
        </div>
      </div>

      {expanded && (
        <div className="border-t border-gray-200 p-4 bg-gray-50">
          {/* Metrics */}
          {experiment.metrics && Object.keys(experiment.metrics).length > 0 && (
            <div className="mb-4">
              <h4 className="text-sm font-medium text-gray-700 mb-2">Key Metrics</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                <MetricCard label="LPIPS" value={experiment.metrics.lpips} target="< 0.35" />
                <MetricCard label="SSIM" value={experiment.metrics.ssim} target="> 0.75" />
                <MetricCard label="Spatial IoU" value={experiment.metrics.spatial_iou} target="> 0.6" />
              </div>
            </div>
          )}

          {/* Sub-experiments */}
          {experiment.subExperiments.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-2">
                Sub-experiments
                {experiment.subExperiments.some(s => s.hasError) && (
                  <span className="ml-2 text-orange-600 text-xs">
                    ({experiment.subExperiments.filter(s => s.hasError).length} with errors)
                  </span>
                )}
              </h4>
              <div className="space-y-2">
                {experiment.subExperiments.map(sub => (
                  <div key={sub.id} className={`bg-white rounded-lg p-3 border ${
                    sub.hasError ? 'border-orange-300' : 'border-gray-200'
                  }`}>
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-medium text-sm flex items-center gap-2">
                        {sub.id}
                        {sub.hasError && (
                          <AlertTriangle size={14} className="text-orange-500" title="Has error" />
                        )}
                      </span>
                      <StatusBadge status={sub.status} />
                    </div>
                    {sub.finding && (
                      <p className="text-sm text-gray-600 line-clamp-2">{sub.finding}</p>
                    )}
                    {sub.hasError && sub.error && (
                      <details className="mt-2">
                        <summary className="text-xs text-orange-600 cursor-pointer hover:text-orange-800">
                          View error details
                        </summary>
                        <pre className="mt-1 text-xs bg-orange-50 text-orange-900 p-2 rounded overflow-x-auto max-h-32 whitespace-pre-wrap">
                          {sub.error}
                        </pre>
                      </details>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function AgentCard({ agent }: { agent: Agent }) {
  const timeSince = (date: string) => {
    const seconds = Math.floor((Date.now() - new Date(date).getTime()) / 1000)
    if (seconds < 60) return `${seconds}s ago`
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
    return `${Math.floor(seconds / 3600)}h ago`
  }

  return (
    <div className={`p-3 rounded-lg border ${agent.isActive ? 'bg-green-50 border-green-200' : 'bg-gray-50 border-gray-200'}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity size={16} className={agent.isActive ? 'text-green-600' : 'text-gray-400'} />
          <span className="font-mono text-sm">{agent.id}</span>
        </div>
        <span className={`text-xs ${agent.isActive ? 'text-green-600' : 'text-gray-500'}`}>
          {timeSince(agent.lastUpdate)}
        </span>
      </div>
      <div className="text-sm text-gray-600 mt-1">{agent.experiment}</div>
    </div>
  )
}

function GateCard({ gate, experiments }: { gate: Gate, experiments: Experiment[] }) {
  const gateExps = experiments.filter(e => gate.experiments.includes(e.id))
  const isBlocked = gate.status === 'blocked'

  return (
    <div className={`bg-white rounded-xl p-4 border ${
      isBlocked ? 'border-orange-300 bg-orange-50' : 'border-gray-200'
    }`}>
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold">{gate.name}</h3>
        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
          gate.status === 'passed' ? 'bg-green-100 text-green-800' :
          gate.status === 'blocked' ? 'bg-orange-100 text-orange-800' :
          gate.status === 'in_progress' ? 'bg-yellow-100 text-yellow-800' :
          'bg-gray-100 text-gray-600'
        }`}>
          {gate.status === 'blocked' ? 'Blocked' : gate.progress}
        </span>
      </div>
      {isBlocked && gate.blockers && (
        <div className={`mb-3 text-xs p-2 rounded flex items-start gap-2 ${
          gate.blockerReason === 'errors'
            ? 'text-red-700 bg-red-100'
            : 'text-orange-700 bg-orange-100'
        }`}>
          <AlertTriangle size={14} className="flex-shrink-0 mt-0.5" />
          <span>
            {gate.blockerReason === 'errors'
              ? `Has errors - fix before proceeding: ${gate.blockers.join(', ')}`
              : `Blocked by pivot: ${gate.blockers.join(', ')}`
            }
          </span>
        </div>
      )}
      <div className="space-y-1">
        {gateExps.map(exp => (
          <div key={exp.id} className="flex items-center justify-between text-sm">
            <span className="text-gray-600">{exp.name}</span>
            {exp.status === 'completed' && exp.recommendation === 'proceed' ? (
              <CheckCircle size={16} className="text-green-600" />
            ) : exp.status === 'completed_with_errors' || (exp.status === 'completed' && exp.recommendation === 'investigate') ? (
              <AlertTriangle size={16} className="text-red-500" />
            ) : exp.status === 'completed' && exp.recommendation === 'pivot' ? (
              <XCircle size={16} className="text-orange-500" />
            ) : exp.status === 'running' || exp.status === 'in_progress' ? (
              <Clock size={16} className="text-yellow-600" />
            ) : (
              <div className="w-4 h-4 rounded-full border-2 border-gray-300" />
            )}
          </div>
        ))}
      </div>
      <div className="mt-3 pt-3 border-t border-gray-100 text-xs text-gray-500">
        Unlocks: {gate.unlocks}
      </div>
    </div>
  )
}

function FindingsModal({ experiment, onClose }: { experiment: Experiment, onClose: () => void }) {
  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50" onClick={onClose}>
      <div
        className="bg-white rounded-xl max-w-4xl max-h-[90vh] overflow-hidden flex flex-col"
        onClick={e => e.stopPropagation()}
      >
        <div className="p-4 border-b border-gray-200 flex items-center justify-between">
          <h2 className="text-xl font-semibold">{experiment.name} - Findings</h2>
          <button onClick={onClose} className="p-2 hover:bg-gray-100 rounded-lg">×</button>
        </div>
        <div className="p-6 overflow-y-auto prose">
          {experiment.findings ? (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{experiment.findings}</ReactMarkdown>
          ) : (
            <p className="text-gray-500">No findings available yet.</p>
          )}
        </div>
      </div>
    </div>
  )
}

function App() {
  const [status, setStatus] = useState<Status | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedExperiment, setSelectedExperiment] = useState<Experiment | null>(null)

  const loadStatus = async () => {
    try {
      setLoading(true)
      const res = await fetch('/status.json')
      if (!res.ok) throw new Error('Failed to load status')
      const data = await res.json()
      setStatus(data)
      setError(null)
    } catch (e) {
      setError('Failed to load status. Run `npm run generate` to create status.json')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadStatus()
    // Auto-refresh every 30 seconds
    const interval = setInterval(loadStatus, 30000)
    return () => clearInterval(interval)
  }, [])

  if (loading && !status) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <RefreshCw className="animate-spin text-gray-400" size={32} />
      </div>
    )
  }

  if (error && !status) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <AlertCircle size={48} className="text-red-400 mx-auto mb-4" />
          <p className="text-gray-600">{error}</p>
          <button
            onClick={loadStatus}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  if (!status) return null

  const activeAgents = status.agents.filter(a => a.isActive)

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Foresight Dashboard</h1>
            <p className="text-sm text-gray-500">Research Experiment Tracker</p>
          </div>
          <div className="flex items-center gap-4">
            <a
              href={status.links.wandb}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 text-sm text-gray-600 hover:text-gray-900"
            >
              <BarChart3 size={16} /> W&B <ExternalLink size={12} />
            </a>
            <a
              href={status.links.github}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 text-sm text-gray-600 hover:text-gray-900"
            >
              GitHub <ExternalLink size={12} />
            </a>
            <button
              onClick={loadStatus}
              className="p-2 hover:bg-gray-100 rounded-lg"
              title="Refresh"
            >
              <RefreshCw size={18} className={loading ? 'animate-spin' : ''} />
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        {/* Active Agents */}
        {activeAgents.length > 0 && (
          <section className="mb-8">
            <h2 className="text-lg font-semibold text-gray-900 mb-3 flex items-center gap-2">
              <Activity className="text-green-600" size={20} />
              Active Agents ({activeAgents.length})
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {activeAgents.map(agent => (
                <AgentCard key={agent.id} agent={agent} />
              ))}
            </div>
          </section>
        )}

        {/* Decision Gates */}
        <section className="mb-8">
          <h2 className="text-lg font-semibold text-gray-900 mb-3">Decision Gates</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {status.gates.map(gate => (
              <GateCard key={gate.id} gate={gate} experiments={status.experiments} />
            ))}
          </div>
        </section>

        {/* Experiments */}
        <section>
          <h2 className="text-lg font-semibold text-gray-900 mb-3">Experiments</h2>
          <div className="space-y-3">
            {status.experiments.map(exp => (
              <ExperimentCard
                key={exp.id}
                experiment={exp}
                onViewFindings={setSelectedExperiment}
              />
            ))}
          </div>
        </section>

        {/* Last Updated */}
        <div className="mt-8 text-center text-sm text-gray-400">
          Last updated: {new Date(status.generatedAt).toLocaleString()}
        </div>
      </main>

      {/* Findings Modal */}
      {selectedExperiment && (
        <FindingsModal
          experiment={selectedExperiment}
          onClose={() => setSelectedExperiment(null)}
        />
      )}
    </div>
  )
}

export default App
