class Logger {
  constructor() {
    this.logs = []
    this.maxLogs = 1000 // Keep last 1000 logs in memory
    this.logLevels = {
      DEBUG: 0,
      INFO: 1,
      WARN: 2,
      ERROR: 3
    }
    this.currentLevel = this.logLevels.INFO // Default to INFO level
  }

  setLevel(level) {
    if (this.logLevels[level] !== undefined) {
      this.currentLevel = this.logLevels[level]
    }
  }

  _shouldLog(level) {
    return this.logLevels[level] >= this.currentLevel
  }

  _createLogEntry(level, message, data = {}) {
    const entry = {
      timestamp: new Date().toISOString(),
      level,
      message,
      data,
      userAgent: navigator.userAgent,
      url: window.location.href,
      sessionId: this._getSessionId()
    }

    // Keep logs in memory (limited)
    this.logs.push(entry)
    if (this.logs.length > this.maxLogs) {
      this.logs.shift()
    }

    return entry
  }

  _getSessionId() {
    let sessionId = sessionStorage.getItem('sessionId')
    if (!sessionId) {
      sessionId = Date.now().toString(36) + Math.random().toString(36).substr(2)
      sessionStorage.setItem('sessionId', sessionId)
    }
    return sessionId
  }

  _output(entry) {
    const logMessage = `[${entry.timestamp}] ${entry.level}: ${entry.message}`

    // Console output
    switch (entry.level) {
      case 'DEBUG':
        console.debug(logMessage, entry.data)
        break
      case 'INFO':
        console.info(logMessage, entry.data)
        break
      case 'WARN':
        console.warn(logMessage, entry.data)
        break
      case 'ERROR':
        console.error(logMessage, entry.data)
        break
    }

    // Send important logs to backend
    if (entry.level === 'ERROR' || entry.level === 'WARN') {
      this._sendToBackend(entry)
    }
  }

  async _sendToBackend(entry) {
    try {
      // Send log to backend for centralized logging
      const response = await fetch('/api/logs', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(entry)
      })
      if (!response.ok) {
        console.warn('Failed to send log to backend:', response.status)
      }
    } catch (error) {
      console.warn('Failed to send log to backend:', error)
    }
  }

  debug(message, data = {}) {
    if (this._shouldLog('DEBUG')) {
      const entry = this._createLogEntry('DEBUG', message, data)
      this._output(entry)
    }
  }

  info(message, data = {}) {
    if (this._shouldLog('INFO')) {
      const entry = this._createLogEntry('INFO', message, data)
      this._output(entry)
    }
  }

  warn(message, data = {}) {
    if (this._shouldLog('WARN')) {
      const entry = this._createLogEntry('WARN', message, data)
      this._output(entry)
    }
  }

  error(message, data = {}) {
    if (this._shouldLog('ERROR')) {
      const entry = this._createLogEntry('ERROR', message, data)
      this._output(entry)
    }
  }

  // Specialized logging methods
  logApiCall(method, url, status, duration, data = {}) {
    const level = status >= 400 ? 'ERROR' : status >= 300 ? 'WARN' : 'INFO'
    const message = `API ${method} ${url} - ${status} (${duration}ms)`
    const logData = {
      method,
      url,
      status,
      duration,
      ...data
    }

    switch (level) {
      case 'ERROR':
        this.error(message, logData)
        break
      case 'WARN':
        this.warn(message, logData)
        break
      default:
        this.info(message, logData)
    }
  }

  logUserAction(action, data = {}) {
    this.info(`User Action: ${action}`, {
      action,
      ...data
    })
  }

  logWebSocketMessage(type, data = {}) {
    this.debug(`WebSocket: ${type}`, data)
  }

  logNavigation(from, to, data = {}) {
    this.info(`Navigation: ${from} -> ${to}`, {
      from,
      to,
      ...data
    })
  }

  logError(error, context = {}) {
    this.error(`Error: ${error.message}`, {
      error: {
        name: error.name,
        message: error.message,
        stack: error.stack
      },
      ...context
    })
  }

  // Get recent logs
  getLogs(limit = 100) {
    return this.logs.slice(-limit)
  }

  // Clear logs
  clearLogs() {
    this.logs = []
  }
}

// Global logger instance
const logger = new Logger()

// Set log level from environment or localStorage
const savedLevel = localStorage.getItem('logLevel')
if (savedLevel && logger.logLevels[savedLevel] !== undefined) {
  logger.setLevel(savedLevel)
}

// Export logger and convenience functions
export default logger

// Export convenience functions that safely call logger methods
export const debug = (...args) => logger.debug(...args)
export const info = (...args) => logger.info(...args)
export const warn = (...args) => logger.warn(...args)
export const error = (...args) => logger.error(...args)
export const logApiCall = (...args) => logger.logApiCall(...args)
export const logUserAction = (...args) => logger.logUserAction(...args)
export const logWebSocketMessage = (...args) => logger.logWebSocketMessage(...args)
export const logNavigation = (...args) => logger.logNavigation(...args)
export const logError = (...args) => logger.logError(...args)