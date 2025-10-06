import PropTypes from 'prop-types'
import { createContext, useContext, useEffect, useRef, useState } from 'react'
import ReconnectingWebSocket from 'reconnecting-websocket'
import { logWebSocketMessage } from '../services/logger.js'

const WebSocketContext = createContext()

export function WebSocketProvider({ children }) {
  const [isConnected, setIsConnected] = useState(false)
  const [messages, setMessages] = useState([])
  const ws = useRef(null)

  useEffect(() => {
    // For development, connect directly to backend. For production, use relative URL with proxy
    const isDevelopment = import.meta.env.DEV
    let wsUrl

    if (isDevelopment) {
      // In development, Vite proxy handles this
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      wsUrl = `${protocol}//${window.location.host}/ws`
    } else {
      // In production, connect directly to backend
      wsUrl = 'ws://localhost:8000/ws'
    }

    console.log('Connecting to WebSocket URL:', wsUrl)

    ws.current = new ReconnectingWebSocket(wsUrl, [], {
      maxReconnectionDelay: 10000,
      minReconnectionDelay: 1000,
      reconnectionDelayGrowFactor: 1.3,
      maxRetries: Infinity,
    })

    ws.current.onopen = () => {
      logWebSocketMessage('connection_opened', { url: ws.current.url })
      setIsConnected(true)
    }

    ws.current.onclose = () => {
      logWebSocketMessage('connection_closed', { url: ws.current.url })
      setIsConnected(false)
    }

    ws.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        logWebSocketMessage('message_received', {
          type: data.type,
          messageLength: event.data.length
        })
        setMessages(prev => [...prev, data])
      } catch (error) {
        logWebSocketMessage('message_parse_error', {
          error: error.message,
          rawData: event.data.substring(0, 200)
        })
      }
    }

    ws.current.onerror = (error) => {
      logWebSocketMessage('connection_error', { error: error.message })
    }

    return () => {
      if (ws.current) {
        ws.current.close()
      }
    }
  }, [])

  const sendMessage = (message) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      logWebSocketMessage('message_sent', {
        type: message.type,
        messageLength: JSON.stringify(message).length
      })
      ws.current.send(JSON.stringify(message))
    } else {
      logWebSocketMessage('message_send_failed', {
        type: message.type,
        readyState: ws.current?.readyState
      })
    }
  }

  return (
    <WebSocketContext.Provider value={{ isConnected, messages, sendMessage }}>
      {children}
    </WebSocketContext.Provider>
  )
}

export function useWebSocket() { // eslint-disable-line react-refresh/only-export-components
  const context = useContext(WebSocketContext)
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider')
  }
  return context
}

WebSocketProvider.propTypes = {
  children: PropTypes.node.isRequired,
}