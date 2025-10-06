import PropTypes from 'prop-types'
import { createContext, useContext, useState } from 'react'

const NotificationContext = createContext()

export function NotificationProvider({ children }) {
  const [notifications, setNotifications] = useState([])

  const addNotification = (message, type = 'info', duration = 5000) => {
    const id = Date.now()
    const notification = { id, message, type, duration }

    setNotifications(prev => [...prev, notification])

    if (duration > 0) {
      setTimeout(() => {
        removeNotification(id)
      }, duration)
    }

    return id
  }

  const removeNotification = (id) => {
    setNotifications(prev => prev.filter(n => n.id !== id))
  }

  const showSuccess = (message, duration) => addNotification(message, 'success', duration)
  const showError = (message, duration) => addNotification(message, 'error', duration)
  const showWarning = (message, duration) => addNotification(message, 'warning', duration)
  const showInfo = (message, duration) => addNotification(message, 'info', duration)

  return (
    <NotificationContext.Provider value={{
      notifications,
      addNotification,
      removeNotification,
      showSuccess,
      showError,
      showWarning,
      showInfo
    }}>
      {children}
    </NotificationContext.Provider>
  )
}

export function useNotifications() { // eslint-disable-line react-refresh/only-export-components
  const context = useContext(NotificationContext)
  if (!context) {
    throw new Error('useNotifications must be used within a NotificationProvider')
  }
  return context
}

NotificationProvider.propTypes = {
  children: PropTypes.node.isRequired,
}