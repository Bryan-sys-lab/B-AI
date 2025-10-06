import { useState, useEffect } from 'react'
import { Github, Plus, RefreshCw, ExternalLink, GitBranch, Star, Eye, Download } from 'lucide-react'
import { repositoryAPI, githubAPI } from '../services/api'
import { useNotifications } from '../providers/NotificationProvider'

function RepositoryManagement() {
  const [repositories, setRepositories] = useState([])
  const [githubRepos, setGithubRepos] = useState([])
  const [connectionStatus, setConnectionStatus] = useState(null)
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState('managed')
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [newRepo, setNewRepo] = useState({ name: '', description: '', is_private: false })
  const { showSuccess, showError } = useNotifications()

  useEffect(() => {
    loadData()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const loadData = async () => {
    try {
      await Promise.all([
        loadRepositories(),
        loadGithubConnection(),
        loadGithubRepos()
      ])
    } catch (error) {
      console.error('Failed to load data:', error)
    } finally {
      setLoading(false)
    }
  }

  const loadRepositories = async () => {
    try {
      const response = await repositoryAPI.list()
      setRepositories(response.data)
    } catch (error) {
      showError('Failed to load repositories')
    }
  }

  const loadGithubConnection = async () => {
    try {
      const response = await githubAPI.getConnectionStatus()
      setConnectionStatus(response.data)
    } catch (error) {
      setConnectionStatus({ connected: false })
    }
  }

  const loadGithubRepos = async () => {
    if (!connectionStatus?.connected) return

    try {
      const response = await githubAPI.getUserRepos()
      setGithubRepos(response.data)
    } catch (error) {
      showError('Failed to load GitHub repositories')
    }
  }

  const connectGithub = async () => {
    const token = prompt('Enter your GitHub Personal Access Token:')
    if (!token) return

    try {
      await githubAPI.connect(token)
      showSuccess('GitHub connected successfully')
      loadGithubConnection()
      loadGithubRepos()
    } catch (error) {
      showError('Failed to connect to GitHub')
    }
  }

  const createRepository = async () => {
    try {
      await repositoryAPI.create(newRepo)
      showSuccess('Repository created successfully')
      setNewRepo({ name: '', description: '', is_private: false })
      setShowCreateForm(false)
      loadRepositories()
    } catch (error) {
      showError('Failed to create repository')
    }
  }

  const syncRepository = async (repoId) => {
    try {
      await repositoryAPI.sync(repoId)
      showSuccess('Repository synced successfully')
      loadRepositories()
    } catch (error) {
      showError('Failed to sync repository')
    }
  }

  const createGithubRepo = async () => {
    try {
      await githubAPI.createRepo(newRepo)
      showSuccess('GitHub repository created successfully')
      setNewRepo({ name: '', description: '', is_private: false })
      setShowCreateForm(false)
      loadGithubRepos()
    } catch (error) {
      showError('Failed to create GitHub repository')
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
          Repository Management
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Manage repositories with GitHub integration and deployment
        </p>
      </div>

      {/* GitHub Connection Status */}
      <div className="card mb-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <Github className="text-gray-600 dark:text-gray-400 mr-3" size={24} />
            <div>
              <h3 className="text-lg font-medium">GitHub Integration</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                {connectionStatus?.connected
                  ? `Connected as ${connectionStatus.username}`
                  : 'Not connected to GitHub'
                }
              </p>
            </div>
          </div>
          {!connectionStatus?.connected && (
            <button onClick={connectGithub} className="btn btn-primary">
              Connect GitHub
            </button>
          )}
        </div>
      </div>

      {/* Tabs */}
      <div className="mb-6">
        <div className="flex space-x-2">
          <button
            onClick={() => setActiveTab('managed')}
            className={`px-4 py-2 rounded-lg font-medium ${
              activeTab === 'managed'
                ? 'bg-primary-100 text-primary-700 dark:bg-primary-900 dark:text-primary-300'
                : 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
            }`}
          >
            Managed Repos
          </button>
          <button
            onClick={() => setActiveTab('github')}
            className={`px-4 py-2 rounded-lg font-medium ${
              activeTab === 'github'
                ? 'bg-primary-100 text-primary-700 dark:bg-primary-900 dark:text-primary-300'
                : 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
            }`}
            disabled={!connectionStatus?.connected}
          >
            GitHub Repos
          </button>
        </div>
      </div>

      {/* Create Repository Form */}
      {showCreateForm && (
        <div className="card mb-6">
          <h2 className="text-xl font-semibold mb-4">
            {activeTab === 'managed' ? 'Create Repository' : 'Create GitHub Repository'}
          </h2>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">Repository Name</label>
              <input
                type="text"
                value={newRepo.name}
                onChange={(e) => setNewRepo({...newRepo, name: e.target.value})}
                className="input w-full"
                placeholder="my-awesome-project"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Description</label>
              <textarea
                value={newRepo.description}
                onChange={(e) => setNewRepo({...newRepo, description: e.target.value})}
                className="input w-full"
                rows={3}
                placeholder="Brief description of the repository"
              />
            </div>

            {activeTab === 'github' && (
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="is_private"
                  checked={newRepo.is_private}
                  onChange={(e) => setNewRepo({...newRepo, is_private: e.target.checked})}
                  className="mr-2"
                />
                <label htmlFor="is_private" className="text-sm">Private repository</label>
              </div>
            )}

            <div className="flex space-x-2">
              <button
                onClick={activeTab === 'managed' ? createRepository : createGithubRepo}
                className="btn btn-primary"
              >
                Create Repository
              </button>
              <button
                onClick={() => setShowCreateForm(false)}
                className="btn btn-secondary"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Repository List */}
      <div className="card">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold">
            {activeTab === 'managed' ? 'Managed Repositories' : 'GitHub Repositories'}
          </h2>
          <button
            onClick={() => setShowCreateForm(true)}
            className="btn btn-primary"
          >
            <Plus size={16} className="mr-2" />
            New Repository
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {(activeTab === 'managed' ? repositories : githubRepos).map((repo) => (
            <div key={repo.id} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center">
                  <Github className="text-gray-600 dark:text-gray-400 mr-2" size={20} />
                  <h3 className="font-medium text-gray-900 dark:text-white">
                    {repo.name}
                  </h3>
                </div>
                {repo.private && (
                  <span className="px-2 py-1 text-xs bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200 rounded">
                    Private
                  </span>
                )}
              </div>

              {repo.description && (
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-3 line-clamp-2">
                  {repo.description}
                </p>
              )}

              <div className="flex items-center justify-between text-sm text-gray-600 dark:text-gray-400 mb-4">
                <div className="flex items-center space-x-4">
                  {repo.stargazers_count !== undefined && (
                    <div className="flex items-center">
                      <Star size={14} className="mr-1" />
                      {repo.stargazers_count}
                    </div>
                  )}
                  {repo.watchers_count !== undefined && (
                    <div className="flex items-center">
                      <Eye size={14} className="mr-1" />
                      {repo.watchers_count}
                    </div>
                  )}
                  {repo.forks_count !== undefined && (
                    <div className="flex items-center">
                      <GitBranch size={14} className="mr-1" />
                      {repo.forks_count}
                    </div>
                  )}
                </div>
                {repo.language && (
                  <span className="px-2 py-1 text-xs bg-primary-100 text-primary-800 dark:bg-primary-900 dark:text-primary-200 rounded">
                    {repo.language}
                  </span>
                )}
              </div>

              <div className="flex space-x-2">
                {repo.html_url && (
                  <a
                    href={repo.html_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="btn btn-secondary flex-1 text-sm"
                  >
                    <ExternalLink size={14} className="mr-1" />
                    View
                  </a>
                )}

                {activeTab === 'managed' && (
                  <button
                    onClick={() => syncRepository(repo.id)}
                    className="btn btn-secondary flex-1 text-sm"
                  >
                    <RefreshCw size={14} className="mr-1" />
                    Sync
                  </button>
                )}

                {activeTab === 'github' && (
                  <button
                    onClick={() => {/* TODO: Deploy functionality */}}
                    className="btn btn-primary flex-1 text-sm"
                  >
                    <Download size={14} className="mr-1" />
                    Deploy
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>

        {(activeTab === 'managed' ? repositories : githubRepos).length === 0 && (
          <div className="text-center py-12">
            <Github size={48} className="mx-auto mb-4 text-gray-400" />
            <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
              No repositories found
            </h3>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              {activeTab === 'managed'
                ? 'Create your first repository to get started'
                : 'Connect to GitHub to view your repositories'
              }
            </p>
            {activeTab === 'managed' && (
              <button
                onClick={() => setShowCreateForm(true)}
                className="btn btn-primary"
              >
                <Plus size={16} className="mr-2" />
                Create Repository
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default RepositoryManagement