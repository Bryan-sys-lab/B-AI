import { useState, useEffect } from 'react'
import { BarChart3, Activity, Cpu, HardDrive, RefreshCw, TrendingUp, AlertTriangle, Trash2 } from 'lucide-react'
import { observabilityAPI, systemAPI } from '../services/api'
import { useNotifications } from '../providers/NotificationProvider'

function ObservabilityDashboard() {
  const [metrics, setMetrics] = useState([])
  const [systemStats, setSystemStats] = useState({})
  const [loading, setLoading] = useState(true)
  const [selectedTimeframe, setSelectedTimeframe] = useState('1h')
  const { showSuccess, showError } = useNotifications()

  useEffect(() => {
    loadData()
    // Set up periodic refresh
    const interval = setInterval(loadData, 30000) // Refresh every 30 seconds
    return () => clearInterval(interval)
  }, [selectedTimeframe]) // eslint-disable-line react-hooks/exhaustive-deps

  const loadData = async () => {
    try {
      const [metricsResponse, cacheResponse] = await Promise.all([
        observabilityAPI.getMetrics(),
        systemAPI.getCacheStats()
      ])

      setMetrics(metricsResponse.data)
      setSystemStats(cacheResponse.data)
    } catch (error) {
      showError('Failed to load observability data')
    } finally {
      setLoading(false)
    }
  }

  const handleClearExpiredCache = async () => {
    try {
      const response = await systemAPI.clearExpiredCache()
      showSuccess(`Cleared ${response.data.message}`)
      loadData() // Refresh stats
    } catch (error) {
      showError('Failed to clear expired cache')
    }
  }

  const handleClearAllCache = async () => {
    if (!confirm('Are you sure you want to clear all cache entries? This may affect performance temporarily.')) {
      return
    }

    try {
      const response = await systemAPI.clearAllCache()
      showSuccess(response.data.message)
      loadData() // Refresh stats
    } catch (error) {
      showError('Failed to clear all cache')
    }
  }

  const formatBytes = (bytes) => {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const formatNumber = (num) => {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M'
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K'
    return num.toString()
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
          Observability Dashboard
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Real-time system monitoring and performance metrics
        </p>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <select
            value={selectedTimeframe}
            onChange={(e) => setSelectedTimeframe(e.target.value)}
            className="input"
          >
            <option value="5m">Last 5 minutes</option>
            <option value="1h">Last hour</option>
            <option value="24h">Last 24 hours</option>
            <option value="7d">Last 7 days</option>
          </select>
        </div>
        <button onClick={loadData} className="btn btn-secondary">
          <RefreshCw size={16} className="mr-2" />
          Refresh
        </button>
      </div>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div className="card">
          <div className="flex items-center">
            <Activity className="text-blue-600 dark:text-blue-400 mr-3" size={24} />
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Total Requests</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {formatNumber(systemStats.total_requests || 0)}
              </p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <TrendingUp className="text-green-600 dark:text-green-400 mr-3" size={24} />
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Success Rate</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {systemStats.success_rate ? `${(systemStats.success_rate * 100).toFixed(1)}%` : '0%'}
              </p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <Cpu className="text-purple-600 dark:text-purple-400 mr-3" size={24} />
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Avg Response Time</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {systemStats.avg_response_time ? `${systemStats.avg_response_time.toFixed(0)}ms` : '0ms'}
              </p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <HardDrive className="text-orange-600 dark:text-orange-400 mr-3" size={24} />
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Cache Size</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {formatBytes(systemStats.cache_size || 0)}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Cache Statistics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Cache Performance</h3>
            <div className="flex space-x-2">
              <button
                onClick={handleClearExpiredCache}
                className="btn btn-secondary btn-sm"
                title="Clear expired cache entries"
              >
                <Trash2 size={14} className="mr-1" />
                Clear Expired
              </button>
              <button
                onClick={handleClearAllCache}
                className="btn btn-danger btn-sm"
                title="Clear all cache entries"
              >
                <Trash2 size={14} className="mr-1" />
                Clear All
              </button>
            </div>
          </div>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600 dark:text-gray-400">Cache Hits</span>
              <span className="font-medium">{formatNumber(systemStats.cache_hits || 0)}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600 dark:text-gray-400">Cache Misses</span>
              <span className="font-medium">{formatNumber(systemStats.cache_misses || 0)}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600 dark:text-gray-400">Hit Rate</span>
              <span className="font-medium">
                {systemStats.cache_hits && systemStats.cache_misses
                  ? `${((systemStats.cache_hits / (systemStats.cache_hits + systemStats.cache_misses)) * 100).toFixed(1)}%`
                  : '0%'
                }
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600 dark:text-gray-400">Expired Entries</span>
              <span className="font-medium">{formatNumber(systemStats.expired_entries || 0)}</span>
            </div>
          </div>
        </div>

        <div className="card">
          <h3 className="text-lg font-semibold mb-4">System Health</h3>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600 dark:text-gray-400">Uptime</span>
              <span className="font-medium">{systemStats.uptime || 'N/A'}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600 dark:text-gray-400">Active Connections</span>
              <span className="font-medium">{systemStats.active_connections || 0}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600 dark:text-gray-400">Memory Usage</span>
              <span className="font-medium">{formatBytes(systemStats.memory_usage || 0)}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600 dark:text-gray-400">CPU Usage</span>
              <span className="font-medium">{systemStats.cpu_usage ? `${systemStats.cpu_usage.toFixed(1)}%` : '0%'}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Metrics */}
      <div className="card">
        <h3 className="text-lg font-semibold mb-4">Recent Activity</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-700">
                <th className="text-left py-2 px-4 font-medium">Metric</th>
                <th className="text-left py-2 px-4 font-medium">Value</th>
                <th className="text-left py-2 px-4 font-medium">Timestamp</th>
                <th className="text-left py-2 px-4 font-medium">Status</th>
              </tr>
            </thead>
            <tbody>
              {metrics.slice(0, 10).map((metric, index) => (
                <tr key={index} className="border-b border-gray-100 dark:border-gray-800">
                  <td className="py-2 px-4">{metric.name}</td>
                  <td className="py-2 px-4">{metric.value}</td>
                  <td className="py-2 px-4">{new Date(metric.timestamp).toLocaleString()}</td>
                  <td className="py-2 px-4">
                    <span className={`inline-flex px-2 py-1 text-xs font-medium rounded-full ${
                      metric.status === 'healthy' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' :
                      metric.status === 'warning' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' :
                      'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                    }`}>
                      {metric.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {metrics.length === 0 && (
          <div className="text-center py-8 text-gray-500 dark:text-gray-400">
            <BarChart3 size={48} className="mx-auto mb-4 opacity-50" />
            <p>No metrics data available</p>
          </div>
        )}
      </div>

      {/* Alerts Section */}
      {(systemStats.alerts || []).length > 0 && (
        <div className="mt-8">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <AlertTriangle className="text-yellow-600 dark:text-yellow-400 mr-2" size={20} />
            Active Alerts
          </h3>
          <div className="space-y-2">
            {systemStats.alerts.map((alert, index) => (
              <div key={index} className="card border-l-4 border-yellow-500">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium text-gray-900 dark:text-white">{alert.message}</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      {new Date(alert.timestamp).toLocaleString()}
                    </p>
                  </div>
                  <span className="px-2 py-1 text-xs font-medium bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200 rounded">
                    {alert.severity}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default ObservabilityDashboard