import { useState, useEffect } from 'react'
import { Zap, Webhook, Github, Slack, Trello, Briefcase, Plus, Settings, RefreshCw, CheckCircle, AlertCircle } from 'lucide-react'
import { integrationAPI } from '../services/api'
import { useNotifications } from '../providers/NotificationProvider'

function IntegrationConnectors() {
  const [integrations, setIntegrations] = useState([])
  const [loading, setLoading] = useState(true)
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [selectedIntegration, setSelectedIntegration] = useState(null)
  const [formData, setFormData] = useState({
    name: '',
    type: 'webhook',
    config: {}
  })
  const { showSuccess, showError } = useNotifications()

  const integrationTypes = [
    { id: 'webhook', name: 'Webhook', icon: Webhook, description: 'HTTP webhook endpoints' },
    { id: 'github', name: 'GitHub', icon: Github, description: 'GitHub repository integration' },
    { id: 'slack', name: 'Slack', icon: Slack, description: 'Slack notifications' },
    { id: 'trello', name: 'Trello', icon: Trello, description: 'Trello board integration' },
    { id: 'jira', name: 'Jira', icon: Briefcase, description: 'Jira issue tracking' }
  ]

  useEffect(() => {
    loadIntegrations()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const loadIntegrations = async () => {
    try {
      const response = await integrationAPI.list()
      setIntegrations(response.data)
    } catch (error) {
      showError('Failed to load integrations')
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    try {
      if (selectedIntegration) {
        // Update existing integration
        await integrationAPI.update(selectedIntegration.id, formData)
        showSuccess('Integration updated successfully')
      } else {
        // Create new integration
        await integrationAPI.create(formData)
        showSuccess('Integration created successfully')
      }
      resetForm()
      loadIntegrations()
    } catch (error) {
      showError(`Failed to ${selectedIntegration ? 'update' : 'create'} integration`)
    }
  }

  const handleSync = async (integrationId) => {
    try {
      await integrationAPI.sync(integrationId)
      showSuccess('Integration synced successfully')
      loadIntegrations()
    } catch (error) {
      showError('Failed to sync integration')
    }
  }

  const resetForm = () => {
    setFormData({
      name: '',
      type: 'webhook',
      config: {}
    })
    setSelectedIntegration(null)
    setShowCreateForm(false)
  }

  const getIntegrationIcon = (type) => {
    const integrationType = integrationTypes.find(t => t.id === type)
    return integrationType ? integrationType.icon : Zap
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'active': return 'text-green-600 dark:text-green-400'
      case 'inactive': return 'text-gray-600 dark:text-gray-400'
      case 'error': return 'text-red-600 dark:text-red-400'
      default: return 'text-gray-600 dark:text-gray-400'
    }
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'active': return <CheckCircle size={16} />
      case 'inactive': return <AlertCircle size={16} />
      case 'error': return <AlertCircle size={16} />
      default: return <AlertCircle size={16} />
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
          Integration Connectors
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Connect external services and automate workflows with webhooks
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Integration Types */}
        <div className="lg:col-span-1">
          <div className="card">
            <h2 className="text-xl font-semibold mb-4">Available Connectors</h2>

            <div className="space-y-4">
              {integrationTypes.map((type) => {
                const Icon = type.icon
                return (
                  <div
                    key={type.id}
                    className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:border-primary-300 dark:hover:border-primary-600 cursor-pointer transition-colors"
                    onClick={() => {
                      setFormData({...formData, type: type.id})
                      setShowCreateForm(true)
                    }}
                  >
                    <div className="flex items-center mb-2">
                      <Icon className="text-primary-600 dark:text-primary-400 mr-3" size={24} />
                      <h3 className="font-medium text-gray-900 dark:text-white">{type.name}</h3>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">{type.description}</p>
                  </div>
                )
              })}
            </div>

            <div className="mt-6 pt-4 border-t border-gray-200 dark:border-gray-700">
              <button
                onClick={() => setShowCreateForm(true)}
                className="btn btn-secondary w-full"
              >
                <Plus size={16} className="mr-2" />
                Custom Integration
              </button>
            </div>
          </div>
        </div>

        {/* Integration Form / Details */}
        <div className="lg:col-span-2">
          {showCreateForm ? (
            <div className="card">
              <h2 className="text-xl font-semibold mb-4">
                {selectedIntegration ? 'Edit Integration' : 'Create New Integration'}
              </h2>

              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium mb-1">Integration Name</label>
                    <input
                      type="text"
                      value={formData.name}
                      onChange={(e) => setFormData({...formData, name: e.target.value})}
                      className="input w-full"
                      placeholder="My Integration"
                      required
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium mb-1">Type</label>
                    <select
                      value={formData.type}
                      onChange={(e) => setFormData({...formData, type: e.target.value})}
                      className="input w-full"
                    >
                      {integrationTypes.map(type => (
                        <option key={type.id} value={type.id}>{type.name}</option>
                      ))}
                    </select>
                  </div>
                </div>

                {/* Type-specific configuration */}
                {formData.type === 'webhook' && (
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium mb-1">Webhook URL</label>
                      <input
                        type="url"
                        value={formData.config.url || ''}
                        onChange={(e) => setFormData({
                          ...formData,
                          config: {...formData.config, url: e.target.value}
                        })}
                        className="input w-full"
                        placeholder="https://example.com/webhook"
                        required
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium mb-1">HTTP Method</label>
                      <select
                        value={formData.config.method || 'POST'}
                        onChange={(e) => setFormData({
                          ...formData,
                          config: {...formData.config, method: e.target.value}
                        })}
                        className="input w-full"
                      >
                        <option value="POST">POST</option>
                        <option value="PUT">PUT</option>
                        <option value="PATCH">PATCH</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-sm font-medium mb-1">Headers (JSON)</label>
                      <textarea
                        value={JSON.stringify(formData.config.headers || {}, null, 2)}
                        onChange={(e) => {
                          try {
                            const headers = JSON.parse(e.target.value)
                            setFormData({
                              ...formData,
                              config: {...formData.config, headers}
                            })
                          } catch (error) {
                            // Invalid JSON, ignore
                          }
                        }}
                        className="input w-full font-mono text-sm"
                        rows={4}
                        placeholder='{"Authorization": "Bearer token", "Content-Type": "application/json"}'
                      />
                    </div>
                  </div>
                )}

                {formData.type === 'github' && (
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium mb-1">Repository Owner</label>
                      <input
                        type="text"
                        value={formData.config.owner || ''}
                        onChange={(e) => setFormData({
                          ...formData,
                          config: {...formData.config, owner: e.target.value}
                        })}
                        className="input w-full"
                        placeholder="octocat"
                        required
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium mb-1">Repository Name</label>
                      <input
                        type="text"
                        value={formData.config.repo || ''}
                        onChange={(e) => setFormData({
                          ...formData,
                          config: {...formData.config, repo: e.target.value}
                        })}
                        className="input w-full"
                        placeholder="Hello-World"
                        required
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium mb-1">GitHub Token</label>
                      <input
                        type="password"
                        value={formData.config.token || ''}
                        onChange={(e) => setFormData({
                          ...formData,
                          config: {...formData.config, token: e.target.value}
                        })}
                        className="input w-full"
                        placeholder="ghp_..."
                        required
                      />
                    </div>
                  </div>
                )}

                {formData.type === 'slack' && (
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium mb-1">Slack Webhook URL</label>
                      <input
                        type="url"
                        value={formData.config.webhook_url || ''}
                        onChange={(e) => setFormData({
                          ...formData,
                          config: {...formData.config, webhook_url: e.target.value}
                        })}
                        className="input w-full"
                        placeholder="https://hooks.slack.com/services/..."
                        required
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium mb-1">Channel</label>
                      <input
                        type="text"
                        value={formData.config.channel || ''}
                        onChange={(e) => setFormData({
                          ...formData,
                          config: {...formData.config, channel: e.target.value}
                        })}
                        className="input w-full"
                        placeholder="#general"
                      />
                    </div>
                  </div>
                )}

                <div className="flex space-x-2">
                  <button type="submit" className="btn btn-primary">
                    {selectedIntegration ? 'Update Integration' : 'Create Integration'}
                  </button>
                  <button type="button" onClick={resetForm} className="btn btn-secondary">
                    Cancel
                  </button>
                </div>
              </form>
            </div>
          ) : (
            <div className="card">
              <div className="text-center py-12">
                <Zap size={48} className="mx-auto mb-4 text-gray-400" />
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                  Select an integration type
                </h3>
                <p className="text-gray-600 dark:text-gray-400">
                  Choose a connector from the left panel to configure a new integration
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Active Integrations */}
      <div className="mt-8">
        <div className="card">
          <h2 className="text-xl font-semibold mb-4">Active Integrations</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {integrations.map((integration) => {
              const Icon = getIntegrationIcon(integration.type)
              return (
                <div key={integration.id} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center">
                      <Icon className="text-primary-600 dark:text-primary-400 mr-3" size={24} />
                      <div>
                        <h3 className="font-medium text-gray-900 dark:text-white">
                          {integration.name}
                        </h3>
                        <p className="text-sm text-gray-600 dark:text-gray-400 capitalize">
                          {integration.type}
                        </p>
                      </div>
                    </div>
                    <div className={`flex items-center ${getStatusColor(integration.status)}`}>
                      {getStatusIcon(integration.status)}
                    </div>
                  </div>

                  <div className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                    Created: {new Date(integration.created_at).toLocaleDateString()}
                  </div>

                  <div className="flex space-x-2">
                    <button
                      onClick={() => handleSync(integration.id)}
                      className="btn btn-secondary flex-1 text-sm"
                    >
                      <RefreshCw size={14} className="mr-1" />
                      Sync
                    </button>
                    <button
                      onClick={() => {
                        setSelectedIntegration(integration)
                        setFormData({
                          name: integration.name,
                          type: integration.type,
                          config: integration.config
                        })
                        setShowCreateForm(true)
                      }}
                      className="btn btn-secondary flex-1 text-sm"
                    >
                      <Settings size={14} className="mr-1" />
                      Configure
                    </button>
                  </div>
                </div>
              )
            })}
          </div>

          {integrations.length === 0 && (
            <div className="text-center py-12">
              <Zap size={48} className="mx-auto mb-4 text-gray-400" />
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                No integrations configured
              </h3>
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                Set up your first integration to connect external services
              </p>
              <button
                onClick={() => setShowCreateForm(true)}
                className="btn btn-primary"
              >
                <Plus size={16} className="mr-2" />
                Create Integration
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Integration Templates */}
      <div className="mt-8">
        <div className="card">
          <h2 className="text-xl font-semibold mb-4">Integration Templates</h2>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            Quick-start templates for common integration patterns
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:border-primary-300 dark:hover:border-primary-600 cursor-pointer transition-colors">
              <Webhook className="text-blue-600 dark:text-blue-400 mb-2" size={24} />
              <h3 className="font-medium mb-1">Task Completion Webhook</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Notify external systems when tasks complete
              </p>
            </div>

            <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:border-primary-300 dark:hover:border-primary-600 cursor-pointer transition-colors">
              <Github className="text-gray-600 dark:text-gray-400 mb-2" size={24} />
              <h3 className="font-medium mb-1">GitHub PR Integration</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Sync pull requests with project tasks
              </p>
            </div>

            <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:border-primary-300 dark:hover:border-primary-600 cursor-pointer transition-colors">
              <Slack className="text-purple-600 dark:text-purple-400 mb-2" size={24} />
              <h3 className="font-medium mb-1">Slack Notifications</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Send alerts to Slack channels
              </p>
            </div>

            <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:border-primary-300 dark:hover:border-primary-600 cursor-pointer transition-colors">
              <Briefcase className="text-blue-600 dark:text-blue-400 mb-2" size={24} />
              <h3 className="font-medium mb-1">Jira Issue Sync</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Create Jira issues from tasks
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default IntegrationConnectors