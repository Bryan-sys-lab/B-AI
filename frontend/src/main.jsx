import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'
import { logError } from './services/logger.js'

// Global error handler for unhandled errors
window.addEventListener('error', (event) => {
  logError(new Error(event.message), {
    filename: event.filename,
    lineno: event.lineno,
    colno: event.colno,
    stack: event.error?.stack
  })
})

// Global error handler for unhandled promise rejections
window.addEventListener('unhandledrejection', (event) => {
  logError(new Error(`Unhandled promise rejection: ${event.reason}`), {
    reason: event.reason,
    promise: event.promise
  })
})

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)