import { useState, useEffect } from 'react'
import Navigation from './components/Navigation'
import Footer from './components/Footer'
import ChatInterface from './components/ChatInterface'
import AgentStatusDashboard from './components/AgentStatusDashboard'
import ProviderManagement from './components/ProviderManagement'
import Workspace from './components/Workspace'
import TaskOrchestrator from './components/TaskOrchestrator'
import ProjectManagement from './components/ProjectManagement'
import RepositoryManagement from './components/RepositoryManagement'
import ObservabilityDashboard from './components/ObservabilityDashboard'
import SecurityInterface from './components/SecurityInterface'
import PromptEngineeringStudio from './components/PromptEngineeringStudio'
import CodeIntelligenceTools from './components/CodeIntelligenceTools'
import IntegrationConnectors from './components/IntegrationConnectors'
import LoggingDashboard from './components/LoggingDashboard'
import ThemeProvider from './providers/ThemeProvider'
import { WebSocketProvider } from './providers/WebSocketProvider'
import { NotificationProvider } from './providers/NotificationProvider'
import NotificationContainer from './components/NotificationContainer'
import logger from './services/logger.js'

const views = {
  chat: ChatInterface,
  agents: AgentStatusDashboard,
  providers: ProviderManagement,
  workspace: Workspace,
  orchestrator: TaskOrchestrator,
  projects: ProjectManagement,
  repositories: RepositoryManagement,
  observability: ObservabilityDashboard,
  security: SecurityInterface,
  prompts: PromptEngineeringStudio,
  intelligence: CodeIntelligenceTools,
  integrations: IntegrationConnectors,
  logging: LoggingDashboard,
}

function App() {
  // Initialize currentView from URL hash or default to 'chat'
  const getInitialView = () => {
    const hash = window.location.hash.slice(1) // Remove the '#'
    return hash && views[hash] ? hash : 'chat'
  }

  const [currentView, setCurrentView] = useState(getInitialView)
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

  const CurrentViewComponent = views[currentView]

  // Update URL when view changes
  const handleViewChange = (newView) => {
    setCurrentView(newView)
    window.location.hash = newView
  }

  // Listen for hash changes (back/forward navigation)
  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.slice(1)
      if (hash && views[hash]) {
        setCurrentView(hash)
      }
    }

    window.addEventListener('hashchange', handleHashChange)
    return () => window.removeEventListener('hashchange', handleHashChange)
  }, [])

  // Log app initialization
  useEffect(() => {
    logger.info('Aetherium frontend initialized', {
      initialView: currentView,
      userAgent: navigator.userAgent,
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight
      }
    })
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Log view changes
  useEffect(() => {
    logger.info(`View changed to: ${currentView}`, {
      view: currentView,
      timestamp: new Date().toISOString()
    })
  }, [currentView])

  return (
    <ThemeProvider>
      <WebSocketProvider>
        <NotificationProvider>
          <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex flex-col">
            <Navigation
              currentView={currentView}
              onViewChange={handleViewChange}
              isMobileMenuOpen={isMobileMenuOpen}
              setIsMobileMenuOpen={setIsMobileMenuOpen}
            />

            <main className="pt-16 flex-1">
              <div className="px-4 py-6 sm:px-6 lg:px-8">
                <CurrentViewComponent />
              </div>
            </main>

            <Footer />
            <NotificationContainer />
          </div>
        </NotificationProvider>
      </WebSocketProvider>
    </ThemeProvider>
  )
}

export default App