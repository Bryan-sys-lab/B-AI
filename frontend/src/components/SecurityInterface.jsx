import { useState, useEffect } from 'react'
import { Shield, Plus, Edit, Trash2, Play, AlertTriangle, CheckCircle, Clock, RefreshCw } from 'lucide-react'
import { securityAPI } from '../services/api'
import { useNotifications } from '../providers/NotificationProvider'

function SecurityInterface() {
  const [policies, setPolicies] = useState([])
  const [scans, setScans] = useState([])
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState('policies')
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [editingPolicy, setEditingPolicy] = useState(null)
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    rules: '',
    severity: 'medium'
  })
  const { showSuccess, showError } = useNotifications()

  useEffect(() => {
    loadData()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const loadData = async () => {
    try {
      const [policiesResponse, scansResponse] = await Promise.all([
        securityAPI.getPolicies(),
        securityAPI.getScans()
      ])
      setPolicies(policiesResponse.data)
      setScans(scansResponse.data)
    } catch (error) {
      showError('Failed to load security data')
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    try {
      if (editingPolicy) {
        await securityAPI.updatePolicy(editingPolicy.id, formData)
        showSuccess('Policy updated successfully')
      } else {
        await securityAPI.createPolicy(formData)
        showSuccess('Policy created successfully')
      }
      resetForm()
      loadData()
    } catch (error) {
      showError(`Failed to ${editingPolicy ? 'update' : 'create'} policy`)
    }
  }

  const handleEdit = (policy) => {
    setEditingPolicy(policy)
    setFormData({
      name: policy.name,
      description: policy.description,
      rules: policy.rules,
      severity: policy.severity
    })
    setShowCreateForm(true)
  }

  const handleDelete = async (policyId) => {
    if (!confirm('Are you sure you want to delete this policy?')) return

    try {
      await securityAPI.deletePolicy(policyId)
      showSuccess('Policy deleted successfully')
      loadData()
    } catch (error) {
      showError('Failed to delete policy')
    }
  }

  const runScan = async () => {
    try {
      await securityAPI.createScan({ type: 'full', target: 'workspace' })
      showSuccess('Security scan started')
      loadData()
    } catch (error) {
      showError('Failed to start security scan')
    }
  }

  const resetForm = () => {
    setFormData({
      name: '',
      description: '',
      rules: '',
      severity: 'medium'
    })
    setEditingPolicy(null)
    setShowCreateForm(false)
  }

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'critical': return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
      case 'high': return 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200'
      case 'medium': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
      case 'low': return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
    }
  }

  const getScanStatusColor = (status) => {
    switch (status) {
      case 'completed': return 'text-green-600 dark:text-green-400'
      case 'running': return 'text-blue-600 dark:text-blue-400'
      case 'failed': return 'text-red-600 dark:text-red-400'
      default: return 'text-gray-600 dark:text-gray-400'
    }
  }

  const getScanStatusIcon = (status) => {
    switch (status) {
      case 'completed': return <CheckCircle size={16} />
      case 'running': return <RefreshCw size={16} className="animate-spin" />
      case 'failed': return <AlertTriangle size={16} />
      default: return <Clock size={16} />
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
          Security Policies
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Configure security policies and run vulnerability scans
        </p>
      </div>

      {/* Tabs */}
      <div className="mb-6">
        <div className="flex space-x-2">
          <button
            onClick={() => setActiveTab('policies')}
            className={`px-4 py-2 rounded-lg font-medium ${
              activeTab === 'policies'
                ? 'bg-primary-100 text-primary-700 dark:bg-primary-900 dark:text-primary-300'
                : 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
            }`}
          >
            Policies
          </button>
          <button
            onClick={() => setActiveTab('scans')}
            className={`px-4 py-2 rounded-lg font-medium ${
              activeTab === 'scans'
                ? 'bg-primary-100 text-primary-700 dark:bg-primary-900 dark:text-primary-300'
                : 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
            }`}
          >
            Security Scans
          </button>
        </div>
      </div>

      {activeTab === 'policies' && (
        <>
          {/* Create Policy Form */}
          {showCreateForm && (
            <div className="card mb-6">
              <h2 className="text-xl font-semibold mb-4">
                {editingPolicy ? 'Edit Policy' : 'Create Security Policy'}
              </h2>

              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium mb-1">Policy Name</label>
                    <input
                      type="text"
                      value={formData.name}
                      onChange={(e) => setFormData({...formData, name: e.target.value})}
                      className="input w-full"
                      required
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium mb-1">Severity</label>
                    <select
                      value={formData.severity}
                      onChange={(e) => setFormData({...formData, severity: e.target.value})}
                      className="input w-full"
                    >
                      <option value="low">Low</option>
                      <option value="medium">Medium</option>
                      <option value="high">High</option>
                      <option value="critical">Critical</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-1">Description</label>
                  <textarea
                    value={formData.description}
                    onChange={(e) => setFormData({...formData, description: e.target.value})}
                    className="input w-full"
                    rows={3}
                    required
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-1">Rules (JSON)</label>
                  <textarea
                    value={formData.rules}
                    onChange={(e) => setFormData({...formData, rules: e.target.value})}
                    className="input w-full font-mono text-sm"
                    rows={6}
                    placeholder='{"patterns": ["*.key", "*.pem"], "actions": ["block", "alert"]}'
                    required
                  />
                </div>

                <div className="flex space-x-2">
                  <button type="submit" className="btn btn-primary">
                    {editingPolicy ? 'Update Policy' : 'Create Policy'}
                  </button>
                  <button type="button" onClick={resetForm} className="btn btn-secondary">
                    Cancel
                  </button>
                </div>
              </form>
            </div>
          )}

          {/* Policies List */}
          <div className="card">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold">Security Policies</h2>
              <button
                onClick={() => setShowCreateForm(true)}
                className="btn btn-primary"
              >
                <Plus size={16} className="mr-2" />
                New Policy
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {policies.map((policy) => (
                <div key={policy.id} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center">
                      <Shield className="text-primary-600 dark:text-primary-400 mr-2" size={20} />
                      <h3 className="font-medium text-gray-900 dark:text-white">
                        {policy.name}
                      </h3>
                    </div>
                    <span className={`px-2 py-1 text-xs font-medium rounded-full capitalize ${getSeverityColor(policy.severity)}`}>
                      {policy.severity}
                    </span>
                  </div>

                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3 line-clamp-2">
                    {policy.description}
                  </p>

                  <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400 mb-4">
                    <span>Created: {new Date(policy.created_at).toLocaleDateString()}</span>
                    <span>Updated: {new Date(policy.updated_at).toLocaleDateString()}</span>
                  </div>

                  <div className="flex space-x-2">
                    <button
                      onClick={() => handleEdit(policy)}
                      className="btn btn-secondary flex-1 text-sm"
                    >
                      <Edit size={14} className="mr-1" />
                      Edit
                    </button>
                    <button
                      onClick={() => handleDelete(policy.id)}
                      className="btn btn-secondary flex-1 text-sm text-red-600 hover:text-red-700"
                    >
                      <Trash2 size={14} className="mr-1" />
                      Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>

            {policies.length === 0 && (
              <div className="text-center py-12">
                <Shield size={48} className="mx-auto mb-4 text-gray-400" />
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                  No security policies
                </h3>
                <p className="text-gray-600 dark:text-gray-400 mb-4">
                  Create your first security policy to protect your workspace
                </p>
                <button
                  onClick={() => setShowCreateForm(true)}
                  className="btn btn-primary"
                >
                  <Plus size={16} className="mr-2" />
                  Create Policy
                </button>
              </div>
            )}
          </div>
        </>
      )}

      {activeTab === 'scans' && (
        <div className="space-y-6">
          {/* Scan Controls */}
          <div className="card">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-xl font-semibold">Security Scans</h2>
                <p className="text-gray-600 dark:text-gray-400">
                  Run vulnerability scans on your workspace and projects
                </p>
              </div>
              <button onClick={runScan} className="btn btn-primary">
                <Play size={16} className="mr-2" />
                Run Security Scan
              </button>
            </div>
          </div>

          {/* Scan Results */}
          <div className="card">
            <h3 className="text-lg font-semibold mb-4">Scan History</h3>

            <div className="space-y-4">
              {scans.map((scan) => (
                <div key={scan.id} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center">
                      <div className={`flex items-center mr-3 ${getScanStatusColor(scan.status)}`}>
                        {getScanStatusIcon(scan.status)}
                      </div>
                      <div>
                        <h4 className="font-medium text-gray-900 dark:text-white">
                          {scan.type} Scan - {scan.target}
                        </h4>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          Started: {new Date(scan.started_at).toLocaleString()}
                        </p>
                      </div>
                    </div>
                    <span className={`px-2 py-1 text-xs font-medium rounded-full capitalize ${
                      scan.status === 'completed' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' :
                      scan.status === 'running' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200' :
                      scan.status === 'failed' ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' :
                      'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
                    }`}>
                      {scan.status}
                    </span>
                  </div>

                  {scan.findings && scan.findings.length > 0 && (
                    <div className="mt-3">
                      <h5 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                        Findings ({scan.findings.length})
                      </h5>
                      <div className="space-y-2">
                        {scan.findings.slice(0, 3).map((finding, index) => (
                          <div key={index} className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-800 rounded">
                            <span className="text-sm">{finding.description}</span>
                            <span className={`px-2 py-1 text-xs font-medium rounded ${getSeverityColor(finding.severity)}`}>
                              {finding.severity}
                            </span>
                          </div>
                        ))}
                        {scan.findings.length > 3 && (
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            ... and {scan.findings.length - 3} more findings
                          </p>
                        )}
                      </div>
                    </div>
                  )}

                  {scan.completed_at && (
                    <div className="mt-3 text-sm text-gray-600 dark:text-gray-400">
                      Completed: {new Date(scan.completed_at).toLocaleString()}
                      {scan.duration && ` (${scan.duration}s)`}
                    </div>
                  )}
                </div>
              ))}
            </div>

            {scans.length === 0 && (
              <div className="text-center py-12">
                <Shield size={48} className="mx-auto mb-4 text-gray-400" />
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                  No security scans yet
                </h3>
                <p className="text-gray-600 dark:text-gray-400 mb-4">
                  Run your first security scan to check for vulnerabilities
                </p>
                <button onClick={runScan} className="btn btn-primary">
                  <Play size={16} className="mr-2" />
                  Start Scan
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default SecurityInterface