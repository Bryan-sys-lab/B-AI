import PropTypes from 'prop-types'
import { useTheme } from '../providers/ThemeProvider'
import { logNavigation, logUserAction } from '../services/logger.js'
import {
  MessageSquare,
  Users,
  Settings,
  FolderOpen,
  GitBranch,
  Folder,
  Github,
  BarChart3,
  Shield,
  FileText,
  Brain,
  Zap,
  Menu,
  X,
  Sun,
  Moon,
  FileSearch,
  ChevronLeft,
  ChevronRight
} from 'lucide-react'

const navigationItems = [
  { id: 'chat', label: 'Chat', icon: MessageSquare },
  { id: 'agents', label: 'Agents', icon: Users },
  { id: 'providers', label: 'Providers', icon: Settings },
  { id: 'workspace', label: 'Workspace', icon: FolderOpen },
  { id: 'orchestrator', label: 'Orchestrator', icon: GitBranch },
  { id: 'projects', label: 'Projects', icon: Folder },
  { id: 'repositories', label: 'Repositories', icon: Github },
  { id: 'observability', label: 'Observability', icon: BarChart3 },
  { id: 'security', label: 'Security', icon: Shield },
  { id: 'prompts', label: 'Prompts', icon: FileText },
  { id: 'intelligence', label: 'Intelligence', icon: Brain },
  { id: 'integrations', label: 'Integrations', icon: Zap },
  { id: 'logging', label: 'Logs', icon: FileSearch },
]

function Navigation({ currentView, onViewChange, isMobileMenuOpen, setIsMobileMenuOpen }) {
  const { isDark, toggleTheme } = useTheme()

  return (
    <>
      {/* Top Navigation Bar */}
      <div className="fixed top-0 left-0 right-0 z-50 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 shadow-sm">
        <div className="flex items-center justify-between px-4 py-3">
          {/* Logo/Title */}
          <div className="flex items-center space-x-4">
            <h1 className="text-xl font-bold text-gray-900 dark:text-white">
              Aetherium
            </h1>
          </div>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center space-x-1 overflow-x-auto scrollbar-hide">
            {navigationItems.map((item) => {
              const Icon = item.icon
              const isActive = currentView === item.id

              return (
                <button
                  key={item.id}
                  onClick={() => {
                    logNavigation(currentView, item.id, { trigger: 'navigation_click' })
                    logUserAction('view_change', { from: currentView, to: item.id })
                    onViewChange(item.id)
                  }}
                  className={`
                    flex items-center px-3 py-2 rounded-lg text-sm font-medium transition-colors whitespace-nowrap
                    ${isActive
                      ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300'
                      : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                    }
                  `}
                >
                  <Icon size={16} className="mr-2" />
                  {item.label}
                </button>
              )
            })}
          </nav>

          {/* Right side controls */}
          <div className="flex items-center space-x-2">
            {/* Theme toggle */}
            <button
              onClick={() => {
                logUserAction('theme_toggle', { from: isDark ? 'dark' : 'light', to: isDark ? 'light' : 'dark' })
                toggleTheme()
              }}
              className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              title={`Switch to ${isDark ? 'light' : 'dark'} mode`}
            >
              {isDark ? <Sun size={20} /> : <Moon size={20} />}
            </button>

            {/* Mobile menu button */}
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="md:hidden p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            >
              {isMobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
            </button>
          </div>
        </div>

        {/* Mobile Navigation Menu */}
        {isMobileMenuOpen && (
          <div className="md:hidden border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
            <nav className="px-4 py-2 space-y-1">
              {navigationItems.map((item) => {
                const Icon = item.icon
                const isActive = currentView === item.id

                return (
                  <button
                    key={item.id}
                    onClick={() => {
                      logNavigation(currentView, item.id, { trigger: 'navigation_click' })
                      logUserAction('view_change', { from: currentView, to: item.id })
                      onViewChange(item.id)
                      setIsMobileMenuOpen(false)
                    }}
                    className={`
                      w-full flex items-center px-3 py-3 text-left rounded-lg transition-colors
                      ${isActive
                        ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300'
                        : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                      }
                    `}
                  >
                    <Icon size={20} className="mr-3 flex-shrink-0" />
                    <span className="font-medium">{item.label}</span>
                  </button>
                )
              })}
            </nav>
          </div>
        )}
      </div>

      {/* Mobile overlay */}
      {isMobileMenuOpen && (
        <div
          className="md:hidden fixed inset-0 bg-black bg-opacity-50 z-40"
          onClick={() => setIsMobileMenuOpen(false)}
        />
      )}
    </>
  )
}

Navigation.propTypes = {
  currentView: PropTypes.string.isRequired,
  onViewChange: PropTypes.func.isRequired,
  isMobileMenuOpen: PropTypes.bool.isRequired,
  setIsMobileMenuOpen: PropTypes.func.isRequired,
}

export default Navigation