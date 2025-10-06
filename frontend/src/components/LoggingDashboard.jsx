import { useState, useEffect } from 'react'
import { Download, Trash2, Settings } from 'lucide-react'
import logger from '../services/logger.js'

function LoggingDashboard() {
  const [logs, setLogs] = useState([])
  const [filteredLogs, setFilteredLogs] = useState([])
  const [logLevel, setLogLevel] = useState('INFO')
  const [showDetails, setShowDetails] = useState(false)
  const [autoRefresh, setAutoRefresh] = useState(true)

  useEffect(() => {
    // Load initial logs
    updateLogs()

    // Set up auto-refresh if enabled
    let interval
    if (autoRefresh) {
      interval = setInterval(updateLogs, 2000)
    }

    return () => {
      if (interval) clearInterval(interval)
    }
  }, [autoRefresh])

  useEffect(() => {
    // Filter logs by level
    const filtered = logs.filter(log => {
      const levels = { DEBUG: 0, INFO: 1, WARN: 2, ERROR: 3 }
      return levels[log.level] >= levels[logLevel]
    })
    setFilteredLogs(filtered)
  }, [logs, logLevel])

  const updateLogs = () => {
    setLogs(logger.getLogs(200)) // Get last 200 logs
  }

  const clearLogs = () => {
    logger.clearLogs()
    setLogs([])
    setFilteredLogs([])
  }

  const exportLogs = () => {
    const dataStr = JSON.stringify(filteredLogs, null, 2)
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr)

    const exportFileDefaultName = `frontend-logs-${new Date().toISOString().split('T')[0]}.json`

    const linkElement = document.createElement('a')
    linkElement.setAttribute('href', dataUri)
    linkElement.setAttribute('download', exportFileDefaultName)
    linkElement.click()
  }

  const getLogLevelColor = (level) => {
    switch (level) {
      case 'ERROR': return 'text-red-600 dark:text-red-400'
      case 'WARN': return 'text-yellow-600 dark:text-yellow-400'
      case 'INFO': return 'text-blue-600 dark:text-blue-400'
      case 'DEBUG': return 'text-gray-600 dark:text-gray-400'
      default: return 'text-gray-600 dark:text-gray-400'
    }
  }

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          Frontend Logging Dashboard
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Monitor frontend activities and debug issues
        </p>
      </div>

      {/* Controls */}
      <div className="card mb-6">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center space-x-2">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Log Level:
            </label>
            <select
              value={logLevel}
              onChange={(e) => setLogLevel(e.target.value)}
              className="input text-sm w-24"
            >
              <option value="DEBUG">Debug</option>
              <option value="INFO">Info</option>
              <option value="WARN">Warn</option>
              <option value="ERROR">Error</option>
            </select>
          </div>

          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              id="autoRefresh"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="rounded"
            />
            <label htmlFor="autoRefresh" className="text-sm text-gray-700 dark:text-gray-300">
              Auto-refresh
            </label>
          </div>

          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              id="showDetails"
              checked={showDetails}
              onChange={(e) => setShowDetails(e.target.checked)}
              className="rounded"
            />
            <label htmlFor="showDetails" className="text-sm text-gray-700 dark:text-gray-300">
              Show details
            </label>
          </div>

          <div className="flex space-x-2 ml-auto">
            <button
              onClick={updateLogs}
              className="btn btn-secondary text-sm"
              disabled={autoRefresh}
            >
              <Settings size={16} className="mr-1" />
              Refresh
            </button>
            <button
              onClick={exportLogs}
              className="btn btn-secondary text-sm"
            >
              <Download size={16} className="mr-1" />
              Export
            </button>
            <button
              onClick={clearLogs}
              className="btn btn-secondary text-sm text-red-600 hover:text-red-700"
            >
              <Trash2 size={16} className="mr-1" />
              Clear
            </button>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
        <div className="card">
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {logs.length}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              Total Logs
            </div>
          </div>
        </div>
        <div className="card">
          <div className="text-center">
            <div className="text-2xl font-bold text-red-600 dark:text-red-400">
              {logs.filter(l => l.level === 'ERROR').length}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              Errors
            </div>
          </div>
        </div>
        <div className="card">
          <div className="text-center">
            <div className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">
              {logs.filter(l => l.level === 'WARN').length}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              Warnings
            </div>
          </div>
        </div>
        <div className="card">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
              {logs.filter(l => l.level === 'INFO').length}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              Info
            </div>
          </div>
        </div>
      </div>

      {/* Logs List */}
      <div className="card">
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {filteredLogs.length === 0 ? (
            <div className="text-center text-gray-500 dark:text-gray-400 py-8">
              No logs found for the selected level
            </div>
          ) : (
            filteredLogs.slice().reverse().map((log, index) => (
              <div
                key={`${log.timestamp}-${index}`}
                className="border-b border-gray-200 dark:border-gray-700 last:border-b-0 pb-2 last:pb-0"
              >
                <div className="flex items-start space-x-3">
                  <span className={`text-xs font-mono px-2 py-1 rounded ${getLogLevelColor(log.level)} bg-gray-100 dark:bg-gray-800`}>
                    {log.level}
                  </span>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm text-gray-900 dark:text-white">
                      {log.message}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      {new Date(log.timestamp).toLocaleString()}
                    </div>
                    {showDetails && log.data && Object.keys(log.data).length > 0 && (
                      <details className="mt-2">
                        <summary className="text-xs text-gray-500 dark:text-gray-400 cursor-pointer hover:text-gray-700 dark:hover:text-gray-300">
                          Show details
                        </summary>
                        <pre className="text-xs bg-gray-100 dark:bg-gray-800 p-2 rounded mt-1 overflow-x-auto">
                          {JSON.stringify(log.data, null, 2)}
                        </pre>
                      </details>
                    )}
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}

export default LoggingDashboard