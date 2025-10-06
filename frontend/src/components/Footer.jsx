import { Heart, Github, ExternalLink } from 'lucide-react'

function Footer() {
  const currentYear = new Date().getFullYear()

  return (
    <footer className="fixed bottom-0 left-0 right-0 z-10 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700">
      <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
          {/* Left side - Branding */}
          <div className="flex items-center space-x-2 text-sm text-gray-600 dark:text-gray-400">
            <span>© {currentYear} Aetherium</span>
            <span className="hidden sm:inline">•</span>
            <span className="hidden sm:inline">Powered by NOVA tech</span>
          </div>

          {/* Center - Links */}
          <div className="flex items-center space-x-6">
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center space-x-1 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 transition-colors"
            >
              <Github size={16} />
              <span className="hidden sm:inline">GitHub</span>
            </a>

            <a
              href="#"
              className="flex items-center space-x-1 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 transition-colors"
            >
              <ExternalLink size={16} />
              <span className="hidden sm:inline">Documentation</span>
            </a>
          </div>

          {/* Right side - Made with love */}
          <div className="flex items-center space-x-1 text-sm text-gray-600 dark:text-gray-400">
            <span>Made with</span>
            <Heart size={14} className="text-red-500 fill-current" />
            <span>by NOVA tech</span>
          </div>
        </div>

        {/* Bottom row - Additional info */}
        <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
          <div className="flex flex-col sm:flex-row justify-between items-center space-y-2 sm:space-y-0">
            <div className="text-xs text-gray-500 dark:text-gray-500">
              Aetherium is an AI-powered development assistant for modern software engineering.
            </div>
            <div className="flex items-center space-x-4 text-xs text-gray-500 dark:text-gray-500">
              <span>v1.0.0</span>
              <span>•</span>
              <span>Built with React & FastAPI</span>
            </div>
          </div>
        </div>
      </div>
    </footer>
  )
}

export default Footer