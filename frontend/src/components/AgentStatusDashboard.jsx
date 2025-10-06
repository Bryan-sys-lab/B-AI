import { useState, useEffect } from 'react'
import { Activity, AlertCircle, CheckCircle, Clock, Play, Square, RefreshCw, GitBranch } from 'lucide-react'
import { agentAPI, taskAPI } from '../services/api'
import { useNotifications } from '../providers/NotificationProvider'

const agentTypes = [
  'fix_implementation_agent',
  'debugger_agent',
  'review_agent',
  'testing_agent',
  'security_agent',
  'performance_agent',
  'deployment_agent',
  'monitoring_agent',
  'feedback_agent',
  'comparator_service',
  'architecture',
  'knowledge_agent',
  'memory_agent'
]

function AgentStatusDashboard() {
  const [agents, setAgents] = useState([])
  const [status, setStatus] = useState({})
  const [health, setHealth] = useState({})
  const [tasks, setTasks] = useState([])
  const [loading, setLoading] = useState(true)
  const { showSuccess, showError } = useNotifications()

  useEffect(() => {
    loadAgents()
    loadStatus()
    loadTasks()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const loadAgents = async () => {
    try {
      const response = await agentAPI.list()
      setAgents(response.data)
    } catch (error) {
      showError('Failed to load agents')
    }
  }

  const loadStatus = async () => {
    try {
      const response = await agentAPI.getStatus()
      // Convert array of agents to status and health mappings by type
      const statusMap = {}
      const healthMap = {}
      response.data.forEach(agent => {
        statusMap[agent.type] = agent.status
        healthMap[agent.type] = agent.health
      })
      setStatus(statusMap)
      setHealth(healthMap)
    } catch (error) {
      showError('Failed to load agent status')
    } finally {
      setLoading(false)
    }
  }

  const loadTasks = async () => {
    try {
      const response = await taskAPI.list()
      setTasks(response.data)
    } catch (error) {
      // Don't show error for tasks as it's not critical
      console.warn('Failed to load tasks:', error)
    }
  }

  const handleAgentControl = async (agentId, action) => {
    try {
      await agentAPI.control(agentId, action)
      showSuccess(`Agent ${action} command sent`)
      // Reload status after a short delay
      setTimeout(loadStatus, 1000)
    } catch (error) {
      showError(`Failed to ${action} agent`)
    }
  }

  const getStatusColor = (agentStatus) => {
    switch (agentStatus) {
      case 'running': return 'text-green-600 dark:text-green-400'
      case 'stopped': return 'text-red-600 dark:text-red-400'
      case 'error': return 'text-yellow-600 dark:text-yellow-400'
      default: return 'text-gray-600 dark:text-gray-400'
    }
  }

  const getStatusIcon = (agentStatus) => {
    switch (agentStatus) {
      case 'running': return <CheckCircle size={16} />
      case 'stopped': return <Square size={16} />
      case 'error': return <AlertCircle size={16} />
      default: return <Clock size={16} />
    }
  }

  const getHealthIcon = (agentHealth) => {
    switch (agentHealth) {
      case 'healthy': return <CheckCircle size={16} className="text-green-600 dark:text-green-400" />
      case 'error': return <AlertCircle size={16} className="text-red-600 dark:text-red-400" />
      default: return <Clock size={16} className="text-yellow-600 dark:text-yellow-400" />
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw size={32} className="animate-spin text-primary-600" />
      </div>
    )
  }

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          Agent Status Dashboard
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Monitor and control Aetherium agents across the platform
        </p>
      </div>

      {/* Overall Status */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div className="card">
          <div className="flex items-center">
            <Activity className="text-green-600 dark:text-green-400 mr-3" size={24} />
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Active Agents</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {Object.values(status).filter(s => s === 'running').length}
              </p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <CheckCircle className="text-green-600 dark:text-green-400 mr-3" size={24} />
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Healthy Agents</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {Object.values(health).filter(h => h === 'healthy').length}
              </p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <AlertCircle className="text-yellow-600 dark:text-yellow-400 mr-3" size={24} />
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">With Errors</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {Object.values(status).filter(s => s === 'error').length}
              </p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <Square className="text-red-600 dark:text-red-400 mr-3" size={24} />
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Stopped</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {Object.values(status).filter(s => s === 'stopped').length}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Agent Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {agentTypes.map((agentType) => {
          const agent = agents.find(a => a.type === agentType)
          const agentStatus = status[agentType] || 'unknown'
          const agentHealth = health[agentType] || 'unknown'

          return (
            <div key={agentType} className="card">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white capitalize">
                  {agentType.replace('_', ' ')}
                </h3>
                <div className="flex items-center space-x-2">
                  {getHealthIcon(agentHealth)}
                  {getStatusIcon(agentStatus)}
                </div>
              </div>

              <div className="mb-4">
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Status</p>
                <span className={`inline-flex px-2 py-1 text-xs font-medium rounded-full capitalize
                  ${agentStatus === 'running' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' :
                    agentStatus === 'stopped' ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' :
                    agentStatus === 'error' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' :
                    'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'}`}>
                  {agentStatus}
                </span>
              </div>

              <div className="mb-4">
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Health</p>
                <span className={`inline-flex px-2 py-1 text-xs font-medium rounded-full capitalize
                  ${agentHealth === 'healthy' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' :
                    agentHealth === 'error' ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' :
                    'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'}`}>
                  {agentHealth}
                </span>
              </div>

              {agent && (
                <div className="mb-4">
                  <p className="text-sm text-gray-600 dark:text-gray-400">Version</p>
                  <p className="text-sm font-medium text-gray-900 dark:text-white">{agent.version}</p>
                </div>
              )}

              <div className="flex space-x-2">
                <button
                  onClick={() => handleAgentControl(agentType, 'start')}
                  disabled={agentStatus === 'running'}
                  className="btn btn-secondary flex-1 text-xs disabled:opacity-50"
                >
                  <Play size={14} className="mr-1" />
                  Start
                </button>
                <button
                  onClick={() => handleAgentControl(agentType, 'stop')}
                  disabled={agentStatus === 'stopped'}
                  className="btn btn-secondary flex-1 text-xs disabled:opacity-50"
                >
                  <Square size={14} className="mr-1" />
                  Stop
                </button>
              </div>
            </div>
          )
        })}
      </div>

      {/* Active Workflows */}
      {tasks.filter(task => task.status === 'running').length > 0 && (
        <div className="mt-8">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
              Active Workflows
            </h2>
            <a
              href="/orchestrator"
              className="text-sm text-primary-600 hover:text-primary-700 dark:text-primary-400 dark:hover:text-primary-300"
            >
              View Full Orchestrator →
            </a>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {tasks.filter(task => task.status === 'running').map((task) => (
              <div key={task.id} className="card">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-medium text-gray-900 dark:text-white truncate">
                    {task.description}
                  </h3>
                  <div className="flex items-center text-blue-600 dark:text-blue-400">
                    <RefreshCw size={16} className="animate-spin mr-1" />
                    <span className="text-sm">Running</span>
                  </div>
                </div>

                <div className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                  Type: {task.type}
                </div>

                {task.subtasks && task.subtasks.length > 0 && (
                  <div>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Current Steps:</p>
                    <div className="space-y-1">
                      {task.subtasks.slice(0, 3).map((subtask, index) => (
                        <div key={index} className="flex items-center justify-between text-sm">
                          <span className="truncate flex-1">{subtask.description}</span>
                          <div className={`flex items-center ml-2 ${getStatusColor(subtask.status)}`}>
                            {getStatusIcon(subtask.status)}
                          </div>
                        </div>
                      ))}
                      {task.subtasks.length > 3 && (
                        <div className="text-xs text-gray-500 dark:text-gray-400">
                          +{task.subtasks.length - 3} more steps...
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {task.progress && (
                  <div className="mt-3">
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full"
                        style={{ width: `${task.progress}%` }}
                      />
                    </div>
                    <span className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                      {task.progress}% complete
                    </span>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Refresh Button */}
      <div className="mt-8 text-center">
        <button
          onClick={() => {
            setLoading(true)
            loadStatus()
            loadTasks()
          }}
          className="btn btn-primary"
        >
          <RefreshCw size={16} className="mr-2" />
          Refresh Status
        </button>
      </div>
    </div>
  )
}

export default AgentStatusDashboard