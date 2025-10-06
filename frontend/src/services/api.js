import axios from 'axios'
import logger from './logger.js'

const api = axios.create({
  baseURL: import.meta.env.DEV ? '/api' : 'http://localhost:8000/api',
  timeout: 30000,
})

// Request interceptor for auth and logging
api.interceptors.request.use(
  (config) => {
    // Add auth headers here if needed
    config.startTime = Date.now()
    logger.debug(`API Request: ${config.method?.toUpperCase()} ${config.url}`, {
      method: config.method,
      url: config.url,
      headers: config.headers,
      data: config.data
    })
    return config
  },
  (error) => {
    logger.error('API Request Error', { error: error.message })
    return Promise.reject(error)
  }
)

// Response interceptor for error handling and logging
api.interceptors.response.use(
  (response) => {
    const duration = Date.now() - response.config.startTime
    logger.logApiCall(
      response.config.method?.toUpperCase(),
      response.config.url,
      response.status,
      duration,
      {
        responseSize: JSON.stringify(response.data).length,
        headers: response.headers
      }
    )
    return response
  },
  (error) => {
    const duration = error.config?.startTime ? Date.now() - error.config.startTime : 0
    const status = error.response?.status || 'NETWORK_ERROR'
    const url = error.config?.url || 'unknown'
    const method = error.config?.method?.toUpperCase() || 'UNKNOWN'

    logger.logApiCall(method, url, status, duration, {
      error: error.message,
      responseData: error.response?.data
    })

    return Promise.reject(error)
  }
)

// Task Management
export const taskAPI = {
  create: (data) => api.post('/tasks', data, { timeout: 120000 }), // 2 minutes for complex tasks
  list: (params) => api.get('/tasks', { params }),
  get: (id) => api.get(`/tasks/${id}`),
  orchestrate: (id) => api.post(`/tasks/${id}/orchestrate`),
}

// Project Management
export const projectAPI = {
  create: (data) => api.post('/projects', data),
  list: () => api.get('/projects'),
  get: (id) => api.get(`/projects/${id}`),
  update: (id, data) => api.put(`/projects/${id}`, data),
  delete: (id) => api.delete(`/projects/${id}`),
}

// Provider Management
export const providerAPI = {
  list: () => api.get('/providers'),
  getMetrics: () => api.get('/providers/metrics'),
  get: (id) => api.get(`/providers/${id}`),
  switch: (id) => api.post(`/providers/switch/${id}`),
}

// Agent Management
export const agentAPI = {
  list: () => api.get('/agents'),
  getStatus: () => api.get('/agents/status'),
  get: (id) => api.get(`/agents/${id}`),
  control: (id, action) => api.post(`/agents/${id}/control`, { action }),
}

// Repository Management
export const repositoryAPI = {
  list: () => api.get('/repositories'),
  create: (data) => api.post('/repositories', data),
  get: (id) => api.get(`/repositories/${id}`),
  getFiles: (id) => api.get(`/repositories/${id}/files`),
  sync: (id) => api.post(`/repositories/${id}/sync`),
  createPR: (id, data) => api.post(`/repositories/${id}/pull-request`, data),
}

// GitHub Integration
export const githubAPI = {
  deploy: (data) => api.post('/github/deploy', data),
  createRepo: (data) => api.post('/github/create-repo', data),
  connect: (token) => api.post('/github/connect', { token }),
  getUserRepos: () => api.get('/github/user/repos'),
  getConnectionStatus: () => api.get('/github/connection/status'),
}

// Security Management
export const securityAPI = {
  getPolicies: () => api.get('/security/policies'),
  createPolicy: (data) => api.post('/security/policies', data),
  updatePolicy: (id, data) => api.patch(`/security/policies/${id}`, data),
  deletePolicy: (id) => api.delete(`/security/policies/${id}`),
  getScans: () => api.get('/security/scans'),
  createScan: (data) => api.post('/security/scans', data),
}

// Observability
export const observabilityAPI = {
  getMetrics: () => api.get('/observability/metrics'),
  createMetric: (data) => api.post('/observability/metrics', data),
}

// Prompt Engineering
export const promptAPI = {
  list: (params) => api.get('/prompts', { params }),
  create: (data) => api.post('/prompts', data),
}

// Intelligence Analysis
export const intelligenceAPI = {
  list: () => api.get('/intelligence/analyses'),
  analyze: (data) => api.post('/intelligence/analyze', data),
}

// Integrations
export const integrationAPI = {
  list: () => api.get('/integrations'),
  create: (data) => api.post('/integrations', data),
  sync: (id) => api.post(`/integrations/${id}/sync`),
}

// Workspace/IDE
export const workspaceAPI = {
  getFiles: () => api.get('/workspace/files'),
  getGeneratedFiles: () => api.get('/workspace/generated-files'),
  getFile: (path) => api.get(`/workspace/files/${path}`),
  getGeneratedFile: (path) => api.get(`/workspace/generated-files/${path}`),
  updateFile: (path, content) => api.put(`/workspace/files/${path}`, { content }),
  createFile: (data) => api.post('/workspace/files', data),
  // Enhanced workspace APIs
  addWorkspace: (config) => api.post('/workspace/workspaces', config),
  getWorkspaceFiles: (workspaceId) => api.get(`/workspace/workspaces/${workspaceId}/files`),
  getWorkspaceFile: (workspaceId, path) => api.get(`/workspace/workspaces/${workspaceId}/files/${path}`),
  updateWorkspaceFile: (workspaceId, path, content) => api.put(`/workspace/workspaces/${workspaceId}/files/${path}`, { content }),
}

// Aetherium Agent API
export const aiAgentAPI = {
  chat: (data) => api.post('/tasks', data),
}

// Search API
export const searchAPI = {
  semantic: (query, workspaceId) => api.post('/workspace/search/semantic', { query, workspace_id: workspaceId }),
}

// VCS API
export const vcsAPI = {
  getDiff: (workspacePath) => api.get('/workspace/vcs/diff', { params: { workspace_path: workspacePath } }),
  commit: (workspacePath, message) => api.post('/workspace/vcs/commit', { workspace_path: workspacePath, message }),
  getCommitSuggestions: (diff) => api.get('/workspace/vcs/commit-suggestions', { params: { diff } }),
}

// Visualization API
export const visualizationAPI = {
  getDependencyGraph: (workspaceId) => api.get(`/workspace/visualization/dependency-graph/${workspaceId}`),
  getCallHierarchy: (workspaceId, symbol) => api.get(`/workspace/visualization/call-hierarchy/${workspaceId}`, { params: { symbol } }),
}

// System Utilities
export const systemAPI = {
  executeShell: (command) => api.post('/shell_exec', { command }),
  executeCode: (code, language) => api.post('/execute_code', { code, language }),
  getCacheStats: () => api.get('/cache/stats'),
  clearExpiredCache: () => api.delete('/cache/expired'),
  clearAllCache: () => api.delete('/cache/all'),
  findSimilarPrompts: (prompt) => api.get('/cache/similar', { params: { prompt } }),
  downloadFile: (filename) => api.get(`/downloads/${filename}`, { responseType: 'blob' }),
  getAbout: () => api.get('/about'),
  getHealth: () => api.get('/health'),
}

// Feedback
export const feedbackAPI = {
  submit: (data) => api.post('/feedback', data),
}

export default api