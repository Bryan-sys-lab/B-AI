import { useState, useEffect } from 'react'
import { Zap, TrendingUp, Clock, DollarSign, RefreshCw, CheckCircle } from 'lucide-react'
import { providerAPI } from '../services/api'
import { useNotifications } from '../providers/NotificationProvider'

function ProviderManagement() {
  const [providers, setProviders] = useState([])
  const [metrics, setMetrics] = useState({})
  const [loading, setLoading] = useState(true)
  const { showSuccess, showError } = useNotifications()

  useEffect(() => {
    loadProviders()
    loadMetrics()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const loadProviders = async () => {
    try {
      const response = await providerAPI.list()
      setProviders(response.data)
    } catch (error) {
      showError('Failed to load providers')
    }
  }

  const loadMetrics = async () => {
    try {
      const response = await providerAPI.getMetrics()
      // Transform array format to object format expected by component
      const metricsArray = response.data
      const transformedMetrics = {
        totalRequests: 0,
        successRate: 0.0,
        avgResponseTime: 0.0,
        totalCost: 0.0,
        providers: {}
      }

      let totalWeightedSuccess = 0
      let totalLatency = 0
      let totalCost = 0
      let totalRequests = 0

      metricsArray.forEach(provider => {
        const providerId = provider.provider_id
        transformedMetrics.providers[providerId] = {
          requests: provider.total_requests,
          successRate: provider.success_rate,
          latency: provider.latency,
          cost: provider.cost_estimate,
          tokensUsed: provider.tokens_used,
          lastUsed: provider.last_used
        }

        totalRequests += provider.total_requests
        totalWeightedSuccess += provider.success_rate * provider.total_requests
        totalLatency += provider.latency * provider.total_requests
        totalCost += provider.cost_estimate
      })

      // Calculate overall metrics
      if (totalRequests > 0) {
        transformedMetrics.totalRequests = totalRequests
        transformedMetrics.successRate = totalWeightedSuccess / totalRequests
        transformedMetrics.avgResponseTime = totalLatency / totalRequests
        transformedMetrics.totalCost = totalCost
      }

      setMetrics(transformedMetrics)
    } catch (error) {
      showError('Failed to load metrics')
    } finally {
      setLoading(false)
    }
  }

  const handleSwitchProvider = async (providerId) => {
    try {
      await providerAPI.switch(providerId)
      showSuccess('Provider switched successfully')
      loadMetrics() // Refresh metrics
    } catch (error) {
      showError('Failed to switch provider')
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
          Provider Management
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Monitor and switch between Aetherium providers
        </p>
      </div>

      {/* Metrics Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div className="card">
          <div className="flex items-center">
            <Zap className="text-blue-600 dark:text-blue-400 mr-3" size={24} />
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Total Requests</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {metrics.totalRequests || 0}
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
                {metrics.successRate ? `${(metrics.successRate * 100).toFixed(1)}%` : '0%'}
              </p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <Clock className="text-yellow-600 dark:text-yellow-400 mr-3" size={24} />
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Avg Response Time</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {metrics.avgResponseTime ? `${metrics.avgResponseTime.toFixed(0)}ms` : '0ms'}
              </p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <DollarSign className="text-purple-600 dark:text-purple-400 mr-3" size={24} />
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Total Cost</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                ${metrics.totalCost ? metrics.totalCost.toFixed(4) : '0.0000'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Provider List */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {providers.map((provider) => {
          const providerMetrics = metrics.providers?.[provider.id] || {}
          const isActive = provider.isActive

          return (
            <div key={provider.id} className="card">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  {provider.name}
                </h3>
                {isActive && (
                  <CheckCircle className="text-green-600 dark:text-green-400" size={20} />
                )}
              </div>

              <div className="space-y-3 mb-4">
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Model</p>
                  <p className="text-sm font-medium text-gray-900 dark:text-white">
                    {provider.model}
                  </p>
                </div>

                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Status</p>
                  <span className={`inline-flex px-2 py-1 text-xs font-medium rounded-full
                    ${provider.status === 'online' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' :
                      'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'}`}>
                    {provider.status}
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-gray-600 dark:text-gray-400">Requests</p>
                    <p className="font-medium text-gray-900 dark:text-white">
                      {providerMetrics.requests || 0}
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-600 dark:text-gray-400">Success Rate</p>
                    <p className="font-medium text-gray-900 dark:text-white">
                      {providerMetrics.successRate ? `${(providerMetrics.successRate * 100).toFixed(1)}%` : '0%'}
                    </p>
                  </div>
                </div>
              </div>

              <button
                onClick={() => handleSwitchProvider(provider.id)}
                disabled={isActive}
                className="btn btn-primary w-full disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isActive ? 'Active Provider' : 'Switch to Provider'}
              </button>
            </div>
          )
        })}
      </div>

      {/* Refresh Button */}
      <div className="mt-8 text-center">
        <button
          onClick={() => {
            setLoading(true)
            loadMetrics()
          }}
          className="btn btn-primary"
        >
          <RefreshCw size={16} className="mr-2" />
          Refresh Metrics
        </button>
      </div>
    </div>
  )
}

export default ProviderManagement