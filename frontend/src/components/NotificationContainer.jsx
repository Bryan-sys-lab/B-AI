import { useNotifications } from '../providers/NotificationProvider'
import { X, CheckCircle, AlertCircle, AlertTriangle, Info } from 'lucide-react'

const notificationStyles = {
  success: {
    bg: 'bg-green-50 dark:bg-green-900/20',
    border: 'border-green-200 dark:border-green-800',
    text: 'text-green-800 dark:text-green-200',
    icon: CheckCircle,
    iconColor: 'text-green-600 dark:text-green-400'
  },
  error: {
    bg: 'bg-red-50 dark:bg-red-900/20',
    border: 'border-red-200 dark:border-red-800',
    text: 'text-red-800 dark:text-red-200',
    icon: AlertCircle,
    iconColor: 'text-red-600 dark:text-red-400'
  },
  warning: {
    bg: 'bg-yellow-50 dark:bg-yellow-900/20',
    border: 'border-yellow-200 dark:border-yellow-800',
    text: 'text-yellow-800 dark:text-yellow-200',
    icon: AlertTriangle,
    iconColor: 'text-yellow-600 dark:text-yellow-400'
  },
  info: {
    bg: 'bg-blue-50 dark:bg-blue-900/20',
    border: 'border-blue-200 dark:border-blue-800',
    text: 'text-blue-800 dark:text-blue-200',
    icon: Info,
    iconColor: 'text-blue-600 dark:text-blue-400'
  }
}

function NotificationContainer() {
  const { notifications, removeNotification } = useNotifications()

  return (
    <div className="fixed top-4 right-4 z-50 space-y-2">
      {notifications.map((notification) => {
        const style = notificationStyles[notification.type] || notificationStyles.info
        const Icon = style.icon

        return (
          <div
            key={notification.id}
            className={`
              max-w-sm w-full p-4 rounded-lg border shadow-lg transition-all duration-300
              ${style.bg} ${style.border} ${style.text}
            `}
          >
            <div className="flex items-start">
              <Icon size={20} className={`mr-3 mt-0.5 flex-shrink-0 ${style.iconColor}`} />
              <div className="flex-1">
                <p className="text-sm font-medium">{notification.message}</p>
              </div>
              <button
                onClick={() => removeNotification(notification.id)}
                className="ml-3 flex-shrink-0 p-1 rounded hover:bg-black hover:bg-opacity-10 transition-colors"
              >
                <X size={16} />
              </button>
            </div>
          </div>
        )
      })}
    </div>
  )
}

export default NotificationContainer