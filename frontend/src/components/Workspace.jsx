import { useState, useEffect, useRef } from 'react'
import {
  Folder, FileText, Terminal, Play, Save, RefreshCw, ChevronRight, ChevronDown,
  Search, GitBranch, Cloud, Bot, Eye, Network, GitCommit, Zap, X, Settings,
  Files, Code, Bug, Puzzle, Menu, MoreHorizontal, Plus, Home, Copy, Download, Loader2
} from 'lucide-react'
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels'
import { workspaceAPI, systemAPI, agentAPI, repositoryAPI, aiAgentAPI, searchAPI, vcsAPI, visualizationAPI, projectAPI } from '../services/api'
import { useNotifications } from '../providers/NotificationProvider'
import { useWebSocket } from '../providers/WebSocketProvider'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism'
import Editor, { DiffEditor } from '@monaco-editor/react'
import { motion, AnimatePresence } from 'framer-motion'

function Workspace() {
  // VS Code Style State
  const [files, setFiles] = useState([])
  const [generatedFiles, setGeneratedFiles] = useState([])
  const [openTabs, setOpenTabs] = useState([]) // Multiple open files like VS Code
  const [activeTabId, setActiveTabId] = useState(null)
  const [fileContents, setFileContents] = useState({}) // Store content for each file
  const [originalContents, setOriginalContents] = useState({}) // Store original content for diff
  const [terminalOutput, setTerminalOutput] = useState('')
  const [terminalCommand, setTerminalCommand] = useState('')
  const [loading, setLoading] = useState(true)
  const [sidebarVisible, setSidebarVisible] = useState(true)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [sidebarView, setSidebarView] = useState('explorer') // explorer, search, git, debug, extensions
  const [panelVisible, setPanelVisible] = useState(true)
  const [panelCollapsed, setPanelCollapsed] = useState(false)
  const [panelView, setPanelView] = useState('terminal') // terminal, output, problems, debug
  const [expandedFolders, setExpandedFolders] = useState(new Set())
  const [workspaces, setWorkspaces] = useState([{ id: 'local', name: 'Local Workspace', type: 'local', path: '/home/su/Aetherium/B1.0' }])
  const [activeWorkspace, setActiveWorkspace] = useState('local')
  const [availableProjects, setAvailableProjects] = useState([])
  const [projectContext, setProjectContext] = useState(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState([])
  const [aiAgentMessages, setAiAgentMessages] = useState([])
  const [aiAgentInput, setAiAgentInput] = useState('')
  const [showDiff, setShowDiff] = useState(false)
  const [isAiThinking, setIsAiThinking] = useState(false)
  const [visualizationData, setVisualizationData] = useState(null)
  const [showVisualization, setShowVisualization] = useState(false)
  const [workspaceFiles, setWorkspaceFiles] = useState([]) // Files/folders added to workspace
  const [fileInputRef, setFileInputRef] = useState(null)
  const [folderInputRef, setFolderInputRef] = useState(null)
  const [showCommandPalette, setShowCommandPalette] = useState(false)
  const [commandPaletteQuery, setCommandPaletteQuery] = useState('')
  const [terminalSessionId, setTerminalSessionId] = useState(null)
  const [terminalWebSocket, setTerminalWebSocket] = useState(null)
  const [terminalConnected, setTerminalConnected] = useState(false)
  const [currentAiTaskId, setCurrentAiTaskId] = useState(null)
  const [codeExecutionResults, setCodeExecutionResults] = useState({})
  const [runningCodeBlocks, setRunningCodeBlocks] = useState(new Set())
  const [exportModal, setExportModal] = useState({ show: false, code: '', language: '', messageIndex: -1, codeBlockIndex: -1 })
  const [exportFilename, setExportFilename] = useState('')
  const [exportPath, setExportPath] = useState('')

  const editorRef = useRef(null)
  const { showSuccess, showError } = useNotifications()
  const { messages } = useWebSocket()

  // Monaco Editor completion provider for Aetherium suggestions
  const setupCompletionProvider = (monaco) => {
    // For now, provide basic completion suggestions
    // In full implementation, this would call Aetherium agent API
    const basicCompletions = {
      javascript: [
        'console.log(', 'function ', 'const ', 'let ', 'var ', 'if (', 'for (', 'while (',
        'try {', 'catch (', 'class ', 'import ', 'export ', 'return ', 'async ', 'await '
      ],
      typescript: [
        'console.log(', 'function ', 'const ', 'let ', 'var ', 'if (', 'for (', 'while (',
        'try {', 'catch (', 'class ', 'interface ', 'type ', 'import ', 'export ', 'return ', 'async ', 'await '
      ],
      python: [
        'print(', 'def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except ', 'import ', 'from ', 'return ', 'self.'
      ],
      java: [
        'System.out.println(', 'public ', 'private ', 'class ', 'if (', 'for (', 'while (', 'try {', 'catch ('
      ]
    };

    Object.keys(basicCompletions).forEach(lang => {
      monaco.languages.registerCompletionItemProvider(lang, {
        provideCompletionItems: (model, position) => {
          const word = model.getWordUntilPosition(position);
          const range = {
            startLineNumber: position.lineNumber,
            endLineNumber: position.lineNumber,
            startColumn: word.startColumn,
            endColumn: word.endColumn
          };

          const suggestions = basicCompletions[lang].map((completion, index) => ({
            label: completion,
            kind: monaco.languages.CompletionItemKind.Snippet,
            insertText: completion,
            range: range,
            sortText: `0${index}`,
            filterText: completion
          }));

          return { suggestions };
        }
      });
    });
  }

  // Load state from localStorage on component mount
  useEffect(() => {
    const savedState = localStorage.getItem('workspaceState')
    if (savedState) {
      try {
        const parsedState = JSON.parse(savedState)
        setAiAgentMessages(parsedState.aiAgentMessages || [])
        setOpenTabs(parsedState.openTabs || [])
        setActiveTabId(parsedState.activeTabId || null)
        setFileContents(parsedState.fileContents || {})
        setOriginalContents(parsedState.originalContents || {})
        setExpandedFolders(new Set(parsedState.expandedFolders || []))
        setSidebarView(parsedState.sidebarView || 'explorer')
        setPanelView(parsedState.panelView || 'terminal')
        setTerminalOutput(parsedState.terminalOutput || '')
        setTerminalCommand(parsedState.terminalCommand || '')
        setCodeExecutionResults(parsedState.codeExecutionResults || {})
        setRunningCodeBlocks(new Set(parsedState.runningCodeBlocks || []))
        setCurrentAiTaskId(parsedState.currentAiTaskId || null)
        setIsAiThinking(parsedState.isAiThinking || false)
        setShowDiff(parsedState.showDiff || false)
        setProjectContext(parsedState.projectContext || null)
        setActiveWorkspace(parsedState.activeWorkspace || 'local')
      } catch (error) {
        console.error('Failed to load workspace state:', error)
      }
    }

    loadFiles()
    loadGeneratedFiles()
    loadAvailableProjects()
    initializeTerminal()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Listen for WebSocket messages to update AI assistant responses
  useEffect(() => {
    if (messages.length > 0) {
      const latestMessage = messages[messages.length - 1]

      if (latestMessage.type === 'output' && latestMessage.task_id === currentAiTaskId) {
        // Parse the output message for AI assistant responses
        try {
          const outputs = JSON.parse(latestMessage.message)
          if (outputs && outputs.length > 0) {
            const output = outputs[0]
            let responseText = ''

            // Handle different output formats
            if (output.explanatory_summary) {
              responseText = output.explanatory_summary
            } else if (output.response) {
              responseText = output.response
            } else if (typeof output === 'string') {
              responseText = output
            } else {
              responseText = JSON.stringify(output, null, 2)
            }

            // Add AI response to messages
            const aiMessage = {
              role: 'assistant',
              content: responseText,
              timestamp: new Date()
            }
            setAiAgentMessages(prev => [...prev, aiMessage])
            setIsAiThinking(false)
            setCurrentAiTaskId(null) // Clear current task
          }
        } catch (error) {
          console.error('Failed to parse AI assistant output:', error)
          setIsAiThinking(false)
          setCurrentAiTaskId(null)
        }
      }
    }
  }, [messages, currentAiTaskId]) // eslint-disable-line react-hooks/exhaustive-deps

  // Save state to localStorage whenever important state changes
  useEffect(() => {
    const stateToSave = {
      aiAgentMessages,
      openTabs,
      activeTabId,
      fileContents,
      originalContents,
      expandedFolders: Array.from(expandedFolders),
      sidebarView,
      panelView,
      terminalOutput,
      terminalCommand,
      codeExecutionResults,
      runningCodeBlocks: Array.from(runningCodeBlocks),
      currentAiTaskId,
      isAiThinking,
      showDiff,
      projectContext,
      activeWorkspace
    }

    try {
      localStorage.setItem('workspaceState', JSON.stringify(stateToSave))
    } catch (error) {
      console.error('Failed to save workspace state:', error)
    }
  }, [
    aiAgentMessages,
    openTabs,
    activeTabId,
    fileContents,
    originalContents,
    expandedFolders,
    sidebarView,
    panelView,
    terminalOutput,
    terminalCommand,
    codeExecutionResults,
    runningCodeBlocks,
    currentAiTaskId,
    isAiThinking,
    showDiff,
    projectContext,
    activeWorkspace
  ])

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.ctrlKey && e.shiftKey && e.key === 'P') {
        e.preventDefault()
        setShowCommandPalette(true)
      }
      if (e.key === 'Escape') {
        setShowCommandPalette(false)
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [])

  const loadFiles = async () => {
    try {
      const response = await workspaceAPI.getFiles()
      setFiles(response.data.files || [])
    } catch (error) {
      showError('Failed to load workspace files')
    }
  }

  const loadGeneratedFiles = async () => {
    try {
      const response = await workspaceAPI.getGeneratedFiles()
      setGeneratedFiles(response.data.files || [])
    } catch (error) {
      showError('Failed to load generated files')
    } finally {
      setLoading(false)
    }
  }

  const openFileInTab = async (filePath) => {
    try {
      // Check if file is already open
      const existingTab = openTabs.find(tab => tab.path === filePath)
      if (existingTab) {
        setActiveTabId(existingTab.id)
        return
      }

      // Load file content
      const response = await workspaceAPI.getFile(filePath)
      const content = response.data.content

      // Create new tab
      const newTab = {
        id: `tab-${Date.now()}`,
        path: filePath,
        name: filePath.split('/').pop(),
        isDirty: false,
        language: getLanguageFromPath(filePath)
      }

      // Store content
      setFileContents(prev => ({ ...prev, [filePath]: content }))
      setOriginalContents(prev => ({ ...prev, [filePath]: content }))

      // Add tab
      setOpenTabs(prev => [...prev, newTab])
      setActiveTabId(newTab.id)
      setShowDiff(false)

    } catch (error) {
      showError('Failed to load file content')
    }
  }

  const closeTab = (tabId) => {
    const tabToClose = openTabs.find(tab => tab.id === tabId)
    if (!tabToClose) return

    const newTabs = openTabs.filter(tab => tab.id !== tabId)
    setOpenTabs(newTabs)

    // If closing active tab, switch to another tab
    if (activeTabId === tabId) {
      const newActiveTab = newTabs.length > 0 ? newTabs[newTabs.length - 1] : null
      setActiveTabId(newActiveTab?.id || null)
    }
  }

  const closeAllTabs = () => {
    setOpenTabs([])
    setActiveTabId(null)
    setFileContents({})
    setOriginalContents({})
    showSuccess('All tabs closed')
  }

  const clearWorkspaceState = () => {
    localStorage.removeItem('workspaceState')
    // Reset all state to defaults
    setAiAgentMessages([])
    setOpenTabs([])
    setActiveTabId(null)
    setFileContents({})
    setOriginalContents({})
    setExpandedFolders(new Set())
    setSidebarView('explorer')
    setPanelView('terminal')
    setTerminalOutput('')
    setTerminalCommand('')
    setCodeExecutionResults({})
    setRunningCodeBlocks(new Set())
    setCurrentAiTaskId(null)
    setIsAiThinking(false)
    setShowDiff(false)
    setProjectContext(null)
    setActiveWorkspace('local')
    showSuccess('Workspace state cleared')
  }

  const initializeTerminal = async () => {
    try {
      // Create terminal session
      const response = await fetch('/api/terminal/sessions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ cwd: '/home/su/Aetherium/B1.0' })
      })

      if (response.ok) {
        const data = await response.json()
        setTerminalSessionId(data.session_id)

        // Connect directly to workspace service WebSocket
        const ws = new WebSocket(`ws://localhost:8024/ws/terminal/${data.session_id}`)

        ws.onopen = () => {
          setTerminalConnected(true)
          setTerminalOutput(`Terminal connected\n${data.cwd}$ `)
        }

        ws.onmessage = (event) => {
          const data = JSON.parse(event.data)
          if (data.type === 'stdout' || data.type === 'stderr') {
            setTerminalOutput(prev => prev + data.content)
            // Add prompt after output
            if (data.content.trim() && !data.content.includes('$ ')) {
              setTimeout(() => setTerminalOutput(prev => prev + '$ '), 10)
            }
          } else if (data.type === 'error') {
            setTerminalOutput(prev => prev + `Error: ${data.content}\n$ `)
          }
        }

        ws.onclose = () => {
          setTerminalConnected(false)
          setTerminalOutput(prev => prev + '\nTerminal disconnected\n')
        }

        ws.onerror = (error) => {
          console.error('Terminal WebSocket error:', error)
          setTerminalConnected(false)
        }

        setTerminalWebSocket(ws)
      }
    } catch (error) {
      console.error('Failed to initialize terminal:', error)
      setTerminalOutput('Failed to initialize terminal\n')
    }
  }

  const getLanguageFromPath = (filePath) => {
    const ext = filePath.split('.').pop()?.toLowerCase()
    const languageMap = {
      'js': 'javascript', 'jsx': 'javascript', 'ts': 'typescript', 'tsx': 'typescript',
      'py': 'python', 'java': 'java', 'cpp': 'cpp', 'c': 'c', 'cs': 'csharp',
      'php': 'php', 'rb': 'ruby', 'go': 'go', 'rs': 'rust', 'html': 'html',
      'css': 'css', 'scss': 'scss', 'sass': 'sass', 'less': 'less', 'json': 'json',
      'xml': 'xml', 'yaml': 'yaml', 'yml': 'yaml', 'md': 'markdown', 'sql': 'sql'
    }
    return languageMap[ext] || 'plaintext'
  }

  const saveFile = async (filePath = null) => {
    const pathToSave = filePath || (activeTabId ? openTabs.find(tab => tab.id === activeTabId)?.path : null)
    if (!pathToSave) return

    try {
      const content = fileContents[pathToSave] || ''
      await workspaceAPI.updateFile(pathToSave, content)

      // Mark tab as clean
      setOpenTabs(prev => prev.map(tab =>
        tab.path === pathToSave ? { ...tab, isDirty: false } : tab
      ))

      showSuccess('File saved successfully')
    } catch (error) {
      showError('Failed to save file')
    }
  }

  const addFilesToWorkspace = async (files) => {
    const newFiles = Array.from(files).map(file => ({
      id: `file_${Date.now()}_${Math.random()}`,
      name: file.name,
      path: file.webkitRelativePath || file.name,
      type: 'file',
      file: file,
      size: file.size,
      lastModified: file.lastModified
    }))

    setWorkspaceFiles(prev => [...prev, ...newFiles])
    showSuccess(`Added ${newFiles.length} file(s) to workspace`)
  }

  const addFolderToWorkspace = async (files) => {
    // Group files by their directory structure
    const folderStructure = {}

    Array.from(files).forEach(file => {
      const pathParts = file.webkitRelativePath.split('/')
      let current = folderStructure

      pathParts.forEach((part, index) => {
        if (!current[part]) {
          current[part] = {
            id: `item_${Date.now()}_${Math.random()}`,
            name: part,
            type: index === pathParts.length - 1 ? 'file' : 'directory',
            path: pathParts.slice(0, index + 1).join('/'),
            children: index === pathParts.length - 1 ? undefined : {},
            file: index === pathParts.length - 1 ? file : undefined,
            size: index === pathParts.length - 1 ? file.size : undefined,
            lastModified: index === pathParts.length - 1 ? file.lastModified : undefined
          }
        }
        if (index < pathParts.length - 1) {
          current = current[part].children
        }
      })
    })

    // Convert to array format
    const convertToArray = (obj) => {
      return Object.values(obj).map(item => {
        if (item.children) {
          item.children = convertToArray(item.children)
        }
        return item
      })
    }

    const newFolders = convertToArray(folderStructure)
    setWorkspaceFiles(prev => [...prev, ...newFolders])
    showSuccess(`Added folder with ${files.length} file(s) to workspace`)
  }

  const removeFromWorkspace = (itemId) => {
    setWorkspaceFiles(prev => prev.filter(item => item.id !== itemId))
    showSuccess('Removed from workspace')
  }

  const browseDirectory = async (path) => {
    try {
      // Use the filesystem browse API
      const response = await fetch(`/api/filesystem/browse?path=${encodeURIComponent(path)}`)
      const data = await response.json()

      if (response.ok && !data.error) {
        // Store the browsed files in state
        setBrowsedFiles(prev => ({
          ...prev,
          [path]: data.files
        }))

        // Show success message with count
        const fileCount = data.files.filter(f => f.type === 'file').length
        const dirCount = data.files.filter(f => f.type === 'directory').length
        showSuccess(`Found ${fileCount} files and ${dirCount} directories in ${path}`)

        // Log the results for debugging
        console.log('Directory contents:', data.files)
      } else {
        showError(data.error || 'Failed to browse directory')
      }
    } catch (error) {
      showError('Failed to browse directory: ' + error.message)
    }
  }

  const createNewFile = async (basePath = '') => {
    const fileName = prompt('Enter file name:')
    if (!fileName) return

    try {
      const fullPath = basePath ? `${basePath}/${fileName}` : fileName
      const response = await fetch('/api/workspace/files', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          path: basePath,
          filename: fileName,
          content: ''
        })
      })

      if (response.ok) {
        const result = await response.json()
        showSuccess(`File "${fileName}" created successfully`)

        // Refresh the workspace files
        loadFiles()

        // Open the new file in editor
        openFileInTab(result.path)
      } else {
        showError('Failed to create file')
      }
    } catch (error) {
      showError('Failed to create file: ' + error.message)
    }
  }


  const executeCommand = async () => {
    if (!terminalCommand.trim()) return

    const command = terminalCommand.trim()
    setTerminalOutput(prev => prev + `\n$ ${command}\n`)
    setTerminalCommand('')

    if (terminalWebSocket && terminalWebSocket.readyState === WebSocket.OPEN) {
      // Send command via WebSocket
      terminalWebSocket.send(JSON.stringify({
        type: 'input',
        content: command
      }))
    } else {
      setTerminalOutput(prev => prev + 'Terminal not connected. Please refresh the page.\n$ ')
    }
  }

  const toggleFolder = (path) => {
    const newExpanded = new Set(expandedFolders)
    if (newExpanded.has(path)) {
      newExpanded.delete(path)
    } else {
      newExpanded.add(path)
    }
    setExpandedFolders(newExpanded)
  }

  const collapseAllFolders = () => {
    setExpandedFolders(new Set())
    showSuccess('All folders collapsed')
  }

  const expandAllFolders = () => {
    // Expand all folders in the current file tree
    const newExpanded = new Set()
    const expandRecursive = (items, prefix = '') => {
      items.forEach(item => {
        if (item.type === 'directory') {
          const fullPath = prefix ? `${prefix}/${item.name}` : item.name
          newExpanded.add(fullPath)
          if (item.children) {
            expandRecursive(item.children, fullPath)
          }
        }
      })
    }
    expandRecursive(files.files || [])
    setExpandedFolders(newExpanded)
    showSuccess('All folders expanded')
  }

  const addWorkspace = async (type, config) => {
    try {
      let newWorkspace
      if (type === 'git') {
        newWorkspace = { id: `git-${Date.now()}`, name: config.name, type: 'git', repoUrl: config.repoUrl, branch: config.branch }
      } else if (type === 'cloud') {
        newWorkspace = { id: `cloud-${Date.now()}`, name: config.name, type: 'cloud', provider: config.provider, path: config.path }
      }
      setWorkspaces([...workspaces, newWorkspace])
      showSuccess('Workspace added successfully')
    } catch (error) {
      showError('Failed to add workspace')
    }
  }

  const performSearch = async (query, type = 'text') => {
    try {
      if (type === 'semantic') {
        // Use semantic search API
        const response = await searchAPI.semantic(query, activeWorkspace)
        setSearchResults(response.data.results || [])
      } else {
        // Full-text search across workspaces
        const results = []
        // Implement full-text search logic
        setSearchResults(results)
      }
    } catch (error) {
      showError('Search failed')
    }
  }

  const sendAiMessage = async () => {
    if (!aiAgentInput.trim()) return

    const userMessage = { role: 'user', content: aiAgentInput, timestamp: new Date() }
    setAiAgentMessages([...aiAgentMessages, userMessage])
    setAiAgentInput('')
    setIsAiThinking(true)

    try {
      // Create a task for Aetherium chat
      const taskData = {
        description: aiAgentInput,
        context: {
          type: 'ai_chat',
          activeFile: activeTabId ? openTabs.find(tab => tab.id === activeTabId)?.path : null,
          workspace: activeWorkspace
        }
      }

      const response = await fetch('/api/tasks', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(taskData)
      })

      if (response.ok) {
        const taskResult = await response.json()
        setCurrentAiTaskId(taskResult.task_id) // Track the task ID for WebSocket responses
      } else {
        throw new Error('Failed to create Aetherium task')
      }
    } catch (error) {
      showError('Aetherium agent error: ' + error.message)
      setIsAiThinking(false)
      setCurrentAiTaskId(null)
    }
  }

  const applyAiSuggestion = (suggestion) => {
    if (suggestion.type === 'code_change') {
      setFileContent(suggestion.newContent)
      setShowDiff(true)
    }
  }

  const generateVisualization = async (type) => {
    try {
      setIsAiThinking(true)

      if (type === 'dependency') {
        // Generate dependency graph using Aetherium
        const context = projectContext ? `Project: ${projectContext.project.name}` : 'Current workspace'
        const response = await aiAgentAPI.chat({
          workspace_id: activeWorkspace,
          message: `Generate a dependency graph for ${context}. Analyze the codebase and show relationships between modules, files, and external dependencies.`,
          context: { visualization_type: 'dependency' }
        })

        setVisualizationData({
          type: 'dependency',
          data: response.data.response,
          generated: true
        })

      } else if (type === 'call_hierarchy') {
        // Generate call hierarchy using Aetherium
        const activeTab = openTabs.find(tab => tab.id === activeTabId)
        const fileName = activeTab ? activeTab.name : 'current file'

        const response = await aiAgentAPI.chat({
          workspace_id: activeWorkspace,
          message: `Generate a call hierarchy for ${fileName}. Analyze the functions, methods, and their calling relationships.`,
          context: {
            visualization_type: 'call_hierarchy',
            active_file: activeTab?.path
          }
        })

        setVisualizationData({
          type: 'call_hierarchy',
          data: response.data.response,
          generated: true
        })
      }

      setShowVisualization(true)
    } catch (error) {
      showError('Failed to generate visualization')
    } finally {
      setIsAiThinking(false)
    }
  }

  const loadAvailableProjects = async () => {
    try {
      // Load projects from API
      const response = await fetch('/api/projects')
      const data = await response.json()
      setAvailableProjects(data || [])
    } catch (error) {
      console.error('Failed to load projects:', error)
    }
  }

  const openProjectInWorkspace = async (projectId) => {
    try {
      // For now, just switch to the project context
      // In full implementation, this would create a dedicated workspace
      setActiveWorkspace(`project-${projectId}`)

      // Load project context for Aetherium
      await loadProjectContext(projectId)

      showSuccess(`Switched to project workspace`)
    } catch (error) {
      showError('Failed to open project workspace')
    }
  }

  const loadProjectContext = async (projectId) => {
    try {
      const response = await fetch(`/api/projects/${projectId}`)
      const data = await response.json()
      setProjectContext(data)
    } catch (error) {
      console.error('Failed to load project context:', error)
    }
  }

  const chatAboutProject = async () => {
    if (!aiAgentInput.trim()) return

    const enhancedMessage = projectContext
      ? `About project "${projectContext.project.name}": ${aiAgentInput}`
      : aiAgentInput

    const userMessage = { role: 'user', content: enhancedMessage, timestamp: new Date() }
    setAiAgentMessages([...aiAgentMessages, userMessage])
    setAiAgentInput('')
    setIsAiThinking(true)

    try {
      // Create a task for Aetherium chat
      const taskData = {
        description: enhancedMessage,
        context: {
          type: 'ai_chat',
          project: projectContext,
          activeFile: activeTabId ? openTabs.find(tab => tab.id === activeTabId)?.path : null,
          workspace: activeWorkspace
        }
      }

      const response = await fetch('/api/tasks', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(taskData)
      })

      if (response.ok) {
        const taskResult = await response.json()
        setCurrentAiTaskId(taskResult.task_id) // Track the task ID for WebSocket responses
      } else {
        throw new Error('Failed to create Aetherium task')
      }
    } catch (error) {
      showError('Aetherium agent error: ' + error.message)
      setIsAiThinking(false)
      setCurrentAiTaskId(null)
    }
  }

  // Copy functionality
  const copyToClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(text)
    } catch (err) {
      console.error('Failed to copy text: ', err)
    }
  }

  const runCode = async (code, language, messageIndex, codeBlockIndex) => {
    const codeBlockId = `${messageIndex}-${codeBlockIndex}`
    setRunningCodeBlocks(prev => new Set(prev).add(codeBlockId))

    try {
      const response = await systemAPI.executeCode(code, language)
      setCodeExecutionResults(prev => ({
        ...prev,
        [codeBlockId]: {
          stdout: response.data.stdout || '',
          stderr: response.data.stderr || '',
          exit_code: response.data.exit_code || 0
        }
      }))
    } catch (error) {
      setCodeExecutionResults(prev => ({
        ...prev,
        [codeBlockId]: {
          stdout: '',
          stderr: error.message || 'Failed to execute code',
          exit_code: 1
        }
      }))
    } finally {
      setRunningCodeBlocks(prev => {
        const newSet = new Set(prev)
        newSet.delete(codeBlockId)
        return newSet
      })
    }
  }

  const openExportModal = (code, language, messageIndex, codeBlockIndex) => {
    // Generate a default filename based on language
    const extension = language === 'python' ? 'py' : language === 'javascript' ? 'js' : language
    const defaultFilename = `generated_code_${Date.now()}.${extension}`

    setExportModal({ show: true, code, language, messageIndex, codeBlockIndex })
    setExportFilename(defaultFilename)
    setExportPath('')
  }

  const closeExportModal = () => {
    setExportModal({ show: false, code: '', language: '', messageIndex: -1, codeBlockIndex: -1 })
    setExportFilename('')
    setExportPath('')
  }

  const exportToWorkspace = async () => {
    if (!exportFilename.trim()) {
      showError('Filename is required')
      return
    }

    try {
      await workspaceAPI.createFile({
        filename: exportFilename.trim(),
        path: exportPath.trim(),
        content: exportModal.code
      })

      showSuccess(`Code exported to workspace as ${exportFilename}`)
      closeExportModal()
    } catch (error) {
      if (error.response?.status === 409) {
        showError('File already exists. Please choose a different filename.')
      } else {
        showError('Failed to export code to workspace')
      }
    }
  }

  // Parse message content to separate text and code blocks
  const parseMessageContent = (content) => {
    const parts = []
    const codeBlockRegex = /```(\w+)?\n?([\s\S]*?)```/g
    let lastIndex = 0
    let match

    while ((match = codeBlockRegex.exec(content)) !== null) {
      // Add text before code block
      if (match.index > lastIndex) {
        const textContent = content.slice(lastIndex, match.index)
        if (textContent.trim()) {
          parts.push({ type: 'text', content: textContent })
        }
      }

      // Add code block
      const language = match[1] || 'text'
      const code = match[2]
      parts.push({ type: 'code', language, content: code })

      lastIndex = match.index + match[0].length
    }

    // Add remaining text
    if (lastIndex < content.length) {
      const remainingText = content.slice(lastIndex)
      if (remainingText.trim()) {
        parts.push({ type: 'text', content: remainingText })
      }
    }

    return parts.length > 0 ? parts : [{ type: 'text', content }]
  }

  // Render message content with code blocks
  const renderMessageContent = (content, messageIndex) => {
    const parts = parseMessageContent(content)

    return parts.map((part, partIndex) => {
      if (part.type === 'code') {
        const codeBlockId = `${messageIndex}-${partIndex}`
        const executionResult = codeExecutionResults[codeBlockId]
        const isRunning = runningCodeBlocks.has(codeBlockId)
        const canRun = ['python', 'py', 'javascript', 'js', 'node'].includes(part.language.toLowerCase())

        return (
          <div key={partIndex} className="relative my-2">
            <div className="flex items-center justify-between bg-gray-800 dark:bg-gray-900 px-3 py-1 rounded-t-md">
              <span className="text-xs text-gray-400 uppercase">{part.language}</span>
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => canRun ? runCode(part.content, part.language, messageIndex, partIndex) : null}
                  disabled={isRunning || !canRun}
                  className={`transition-colors ${canRun ? 'text-gray-400 hover:text-green-400' : 'text-gray-600 cursor-not-allowed'}`}
                  title={canRun ? "Run code" : "Code execution not supported for this language"}
                >
                  {isRunning ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
                </button>
                <button
                  onClick={() => openExportModal(part.content, part.language, messageIndex, partIndex)}
                  className="text-gray-400 hover:text-blue-400 transition-colors"
                  title="Export to workspace"
                >
                  <Download size={14} />
                </button>
                <button
                  onClick={() => copyToClipboard(part.content)}
                  className="text-gray-400 hover:text-white transition-colors"
                  title="Copy code"
                >
                  <Copy size={14} />
                </button>
              </div>
            </div>
            <SyntaxHighlighter
              language={part.language}
              style={document.documentElement.classList.contains('dark') ? oneDark : oneLight}
              customStyle={{
                margin: 0,
                borderRadius: executionResult ? '0' : '0 0 6px 6px',
                fontSize: '14px'
              }}
            >
              {part.content}
            </SyntaxHighlighter>
            {executionResult && (
              <div className="bg-gray-900 dark:bg-gray-950 border-t border-gray-700 rounded-b-md p-3">
                <div className="flex items-center space-x-2 mb-2">
                  <Terminal size={14} className="text-gray-400" />
                  <span className="text-xs text-gray-400 uppercase">Output</span>
                  <span className={`text-xs px-2 py-1 rounded ${
                    executionResult.exit_code === 0
                      ? 'bg-green-900 text-green-300'
                      : 'bg-red-900 text-red-300'
                  }`}>
                    Exit Code: {executionResult.exit_code}
                  </span>
                </div>
                {executionResult.stdout && (
                  <pre className="text-sm text-gray-300 whitespace-pre-wrap mb-2">
                    {executionResult.stdout}
                  </pre>
                )}
                {executionResult.stderr && (
                  <pre className="text-sm text-red-400 whitespace-pre-wrap">
                    {executionResult.stderr}
                  </pre>
                )}
              </div>
            )}
          </div>
        )
      } else {
        return (
          <span key={partIndex} className="whitespace-pre-wrap">
            {part.content}
          </span>
        )
      }
    })
  }

  const renderWorkspaceItem = (item, prefix = '', index = 0) => {
    const fullPath = prefix ? `${prefix}/${item.name}` : item.name
    const isExpanded = expandedFolders.has(item.id)

    if (item.type === 'directory') {
      return (
        <div key={`${item.id}-${index}`}>
          <div className="relative group">
            <div
              className="flex items-center py-1 px-2 hover:bg-gray-200 dark:hover:bg-gray-700 cursor-pointer rounded"
              onClick={() => toggleFolder(item.id)}
            >
              {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
              <Folder size={14} className="ml-1 mr-2 text-blue-500" />
              <span className="text-sm flex-1">{item.name}</span>
            </div>
            {/* Directory actions */}
            <div className="absolute right-1 top-1 opacity-0 group-hover:opacity-100 flex space-x-1">
              <button
                onClick={(e) => { e.stopPropagation(); removeFromWorkspace(item.id); }}
                className="p-0.5 hover:bg-red-300 dark:hover:bg-red-600 rounded text-xs"
                title="Remove from Workspace"
              >
                <X size={10} />
              </button>
            </div>
          </div>
          {isExpanded && item.children && (
            <div className="ml-4">
              {item.children.map((child, childIndex) => renderWorkspaceItem(child, fullPath, childIndex))}
            </div>
          )}
        </div>
      )
    } else {
      return (
        <div
          key={`${item.id}-${index}`}
          className="flex items-center py-1 px-2 hover:bg-gray-200 dark:hover:bg-gray-700 cursor-pointer rounded group"
        >
          <FileText size={14} className="ml-5 mr-2 text-gray-500" />
          <span
            className="text-sm flex-1 cursor-pointer"
            onClick={() => openWorkspaceFile(item)}
          >
            {item.name}
          </span>
          <div className="opacity-0 group-hover:opacity-100 flex space-x-1">
            <button
              onClick={(e) => { e.stopPropagation(); removeFromWorkspace(item.id); }}
              className="p-0.5 hover:bg-red-300 dark:hover:bg-red-600 rounded text-xs"
              title="Remove from Workspace"
            >
              <X size={10} />
            </button>
          </div>
        </div>
      )
    }
  }

  const openWorkspaceFile = async (item) => {
    try {
      // Check if file is already open
      const existingTab = openTabs.find(tab => tab.path === item.path)
      if (existingTab) {
        setActiveTabId(existingTab.id)
        return
      }

      // Read file content
      let content = ''
      if (item.file) {
        content = await item.file.text()
      } else {
        // Fallback for files without File object
        content = '// Unable to read file content'
      }

      // Create new tab
      const newTab = {
        id: `tab-${Date.now()}`,
        path: item.path,
        name: item.name,
        isDirty: false,
        language: getLanguageFromPath(item.name),
        workspaceFile: true // Mark as workspace file
      }

      // Store content
      setFileContents(prev => ({ ...prev, [item.path]: content }))
      setOriginalContents(prev => ({ ...prev, [item.path]: content }))

      // Add tab
      setOpenTabs(prev => [...prev, newTab])
      setActiveTabId(newTab.id)
      setShowDiff(false)

    } catch (error) {
      console.error('Failed to open workspace file:', error)
      showError('Failed to open workspace file: ' + error.message)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw size={32} className="animate-spin text-primary-600" />
      </div>
    )
  }

  return (
    <div className="h-screen flex flex-col bg-[#1e1e1e]">
      {/* Command Palette Overlay */}
      <AnimatePresence>
        {showCommandPalette && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-start justify-center pt-20 z-50"
            onClick={() => setShowCommandPalette(false)}
          >
            <motion.div
              className="bg-[#252526] border border-gray-600 rounded-lg shadow-xl w-96 max-w-lg"
              onClick={(e) => e.stopPropagation()}
              initial={{ y: -20 }}
              animate={{ y: 0 }}
            >
              <div className="p-4">
                <input
                  type="text"
                  value={commandPaletteQuery}
                  onChange={(e) => setCommandPaletteQuery(e.target.value)}
                  placeholder="Type a command or search..."
                  className="w-full bg-transparent border-none outline-none text-white placeholder-gray-400"
                  autoFocus
                />
              </div>
              <div className="border-t border-gray-600 max-h-64 overflow-y-auto">
                {/* Command suggestions would go here */}
                <div className="p-2 text-gray-400 text-sm">No commands found</div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* VS Code Style Layout */}
      <PanelGroup direction="horizontal" className="flex-1">
        {/* Activity Bar */}
        <Panel defaultSize={5} minSize={3} maxSize={8} className="bg-[#252526] flex flex-col items-center py-2">
          <button
            onClick={() => setSidebarView('explorer')}
            className={`w-10 h-10 flex items-center justify-center mb-1 rounded ${
              sidebarView === 'explorer' ? 'bg-[#37373d] text-[#007acc]' : 'text-[#cccccc] hover:text-white hover:bg-[#2d2d30]'
            }`}
            title="Explorer"
          >
            <Files size={20} />
          </button>
          <button
            onClick={() => setSidebarView('search')}
            className={`w-10 h-10 flex items-center justify-center mb-1 rounded ${
              sidebarView === 'search' ? 'bg-[#37373d] text-[#007acc]' : 'text-[#cccccc] hover:text-white hover:bg-[#2d2d30]'
            }`}
            title="Search"
          >
            <Search size={20} />
          </button>
          <button
            onClick={() => setSidebarView('git')}
            className={`w-10 h-10 flex items-center justify-center mb-1 rounded ${
              sidebarView === 'git' ? 'bg-[#37373d] text-[#007acc]' : 'text-[#cccccc] hover:text-white hover:bg-[#2d2d30]'
            }`}
            title="Source Control"
          >
            <GitBranch size={20} />
          </button>
          <button
            onClick={() => setSidebarView('debug')}
            className={`w-10 h-10 flex items-center justify-center mb-1 rounded ${
              sidebarView === 'debug' ? 'bg-[#37373d] text-[#007acc]' : 'text-[#cccccc] hover:text-white hover:bg-[#2d2d30]'
            }`}
            title="Run and Debug"
          >
            <Bug size={20} />
          </button>
          <button
            onClick={() => setSidebarView('extensions')}
            className={`w-10 h-10 flex items-center justify-center mb-1 rounded ${
              sidebarView === 'extensions' ? 'bg-[#37373d] text-[#007acc]' : 'text-[#cccccc] hover:text-white hover:bg-[#2d2d30]'
            }`}
            title="Extensions"
          >
            <Puzzle size={20} />
          </button>
        </Panel>

        <PanelResizeHandle className="w-px bg-[#3e3e42] hover:bg-[#007acc] transition-colors" />

        {/* Left Sidebar */}
        {sidebarVisible && (
          <Panel
            defaultSize={sidebarCollapsed ? 3 : 20}
            minSize={3}
            maxSize={30}
            className="bg-[#252526] border-r border-[#3e3e42] flex flex-col"
          >
            {/* Sidebar Header */}
            <div className="h-10 border-b border-gray-300 dark:border-gray-700 flex items-center px-3 justify-between">
              {!sidebarCollapsed && (
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  {sidebarView === 'explorer' && 'EXPLORER'}
                  {sidebarView === 'search' && 'SEARCH'}
                  {sidebarView === 'git' && 'SOURCE CONTROL'}
                  {sidebarView === 'debug' && 'RUN AND DEBUG'}
                  {sidebarView === 'extensions' && 'EXTENSIONS'}
                </span>
              )}
              <button
                onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
                className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded text-gray-500 hover:text-gray-300"
                title={sidebarCollapsed ? "Expand Sidebar" : "Collapse Sidebar"}
              >
                <Menu size={14} />
              </button>
            </div>

            {/* Sidebar Content */}
            {!sidebarCollapsed && (
              <div className="flex-1 overflow-hidden">
                {sidebarView === 'explorer' && (
                  <div className="p-2 h-full overflow-y-auto">
                  {/* File System Roots */}
                  <div className="mb-4">
                    <div className="text-xs text-gray-500 uppercase tracking-wide mb-2">Open Editors</div>
                    {openTabs.map(tab => (
                      <div
                        key={tab.id}
                        className={`flex items-center px-2 py-1 text-sm cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-700 rounded ${
                          activeTabId === tab.id ? 'bg-blue-100 dark:bg-blue-900' : ''
                        }`}
                        onClick={() => setActiveTabId(tab.id)}
                      >
                        <FileText size={14} className="mr-2 text-gray-500" />
                        <span className="flex-1 truncate">{tab.name}</span>
                        <button
                          onClick={(e) => { e.stopPropagation(); closeTab(tab.id); }}
                          className="ml-1 p-0.5 hover:bg-gray-300 dark:hover:bg-gray-600 rounded"
                        >
                          <X size={12} />
                        </button>
                      </div>
                    ))}
                  </div>

                  {/* Workspace Files */}
                  <div className="mb-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs text-gray-500 uppercase tracking-wide">Workspace</span>
                      <div className="flex space-x-1">
                        <button
                          onClick={collapseAllFolders}
                          className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded"
                          title="Collapse All Folders"
                        >
                          <ChevronDown size={12} />
                        </button>
                        <button
                          onClick={expandAllFolders}
                          className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded"
                          title="Expand All Folders"
                        >
                          <ChevronRight size={12} />
                        </button>
                        <button
                          onClick={() => createNewFile('')}
                          className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded"
                          title="New File"
                        >
                          <FileText size={12} />
                        </button>
                        <button onClick={loadFiles} className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded">
                          <RefreshCw size={12} />
                        </button>
                      </div>
                    </div>
                    <div className="space-y-1">
                      {(files.files || []).map((item, index) => renderFileTreeItem(item, '', index))}
                    </div>
                  </div>

                  {/* Workspace Files */}
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs text-gray-500 uppercase tracking-wide">Workspace Files</span>
                      <div className="flex space-x-1">
                        <input
                          ref={setFileInputRef}
                          type="file"
                          multiple
                          style={{ display: 'none' }}
                          onChange={(e) => addFilesToWorkspace(e.target.files)}
                        />
                        <input
                          ref={setFolderInputRef}
                          type="file"
                          webkitdirectory=""
                          style={{ display: 'none' }}
                          onChange={(e) => addFolderToWorkspace(e.target.files)}
                        />
                        <button
                          onClick={() => fileInputRef?.click()}
                          className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded"
                          title="Add Files"
                        >
                          <FileText size={12} />
                        </button>
                        <button
                          onClick={() => folderInputRef?.click()}
                          className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded"
                          title="Add Folder"
                        >
                          <Folder size={12} />
                        </button>
                      </div>
                    </div>

                    {/* Workspace files tree */}
                    <div className="space-y-1">
                      {workspaceFiles.length === 0 ? (
                        <div className="text-center py-8 text-gray-500">
                          <Files size={32} className="mx-auto mb-2 opacity-50" />
                          <p className="text-sm">No files in workspace</p>
                          <p className="text-xs">Click the buttons above to add files</p>
                        </div>
                      ) : (
                        workspaceFiles.map((item, index) => renderWorkspaceItem(item, '', index))
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* Other sidebar views can be added here */}
              {sidebarView !== 'explorer' && sidebarView !== 'search' && (
                <div className="p-4 text-center text-gray-500">
                  {sidebarView} view coming soon...
                </div>
              )}
              </div>
            )}
          </Panel>
        )}

        <PanelResizeHandle className="w-px bg-[#3e3e42] hover:bg-[#007acc] transition-colors" />

        {/* Main Editor Area */}
        <Panel defaultSize={55} minSize={30} className="flex flex-col bg-[#1e1e1e]">
          {/* Editor Tabs */}
          {openTabs.length > 0 && (
            <div className="flex bg-gray-100 dark:bg-gray-800 border-b border-gray-300 dark:border-gray-700">
              {openTabs.map(tab => (
                <div
                  key={tab.id}
                  className={`flex items-center px-3 py-2 text-sm cursor-pointer border-r border-gray-300 dark:border-gray-700 min-w-0 max-w-xs ${
                    activeTabId === tab.id
                      ? 'bg-white dark:bg-gray-900 border-b-2 border-blue-500'
                      : 'hover:bg-gray-200 dark:hover:bg-gray-700'
                  }`}
                  onClick={() => setActiveTabId(tab.id)}
                >
                  <FileText size={14} className="mr-2 text-gray-500 flex-shrink-0" />
                  <span className="truncate">{tab.name}</span>
                  {tab.isDirty && <span className="ml-1 w-2 h-2 bg-orange-500 rounded-full flex-shrink-0"></span>}
                  <button
                    onClick={(e) => { e.stopPropagation(); closeTab(tab.id); }}
                    className="ml-2 p-0.5 hover:bg-gray-300 dark:hover:bg-gray-600 rounded flex-shrink-0"
                  >
                    <X size={12} />
                  </button>
                </div>
              ))}
              {/* Close All Button */}
              <div className="flex items-center px-2 py-2 border-r border-gray-300 dark:border-gray-700">
                <button
                  onClick={closeAllTabs}
                  className="p-1 hover:bg-gray-300 dark:hover:bg-gray-600 rounded text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                  title="Close All Tabs"
                >
                  <X size={14} />
                </button>
              </div>
            </div>
          )}

          {/* Editor Content */}
          <div className="flex-1">
            {activeTabId ? (
              (() => {
                const activeTab = openTabs.find(tab => tab.id === activeTabId)
                const filePath = activeTab?.path
                const content = fileContents[filePath] || ''

                return showDiff ? (
                  <DiffEditor
                    height="100%"
                    language={activeTab?.language || 'plaintext'}
                    original={originalContents[filePath] || ''}
                    modified={content}
                    theme="vs-dark"
                    options={{
                      minimap: { enabled: false },
                      fontSize: 14,
                      lineNumbers: 'on',
                      automaticLayout: true,
                    }}
                  />
                ) : (
                  <Editor
                    height="100%"
                    language={activeTab?.language || 'plaintext'}
                    value={content}
                    onChange={(value) => {
                      setFileContents(prev => ({ ...prev, [filePath]: value }))
                      setOpenTabs(prev => prev.map(tab =>
                        tab.path === filePath ? { ...tab, isDirty: true } : tab
                      ))
                    }}
                    theme="vs-dark"
                    onMount={(editor, monaco) => {
                      editorRef.current = editor
                      setupCompletionProvider(monaco)
                    }}
                    options={{
                      minimap: { enabled: false },
                      fontSize: 14,
                      lineNumbers: 'on',
                      roundedSelection: false,
                      scrollBeyondLastLine: false,
                      automaticLayout: true,
                      quickSuggestions: true,
                      suggestOnTriggerCharacters: true,
                      acceptSuggestionOnEnter: 'on',
                    }}
                  />
                )
              })()
            ) : (
              <div className="h-full flex items-center justify-center text-gray-500">
                <div className="text-center">
                  <Code size={48} className="mx-auto mb-4 opacity-50" />
                  <p className="text-lg mb-2">Welcome to Aetherium</p>
                  <p>Open a file from the explorer to start coding</p>
                </div>
              </div>
            )}
          </div>
        </Panel>

        <PanelResizeHandle className="w-px bg-[#3e3e42] hover:bg-[#007acc] transition-colors" />

        {/* Right Sidebar (Aetherium Assistant) */}
        <Panel defaultSize={20} minSize={15} maxSize={30} className="bg-[#252526] border-l border-[#3e3e42] flex flex-col">
          <div className="h-10 border-b border-gray-300 dark:border-gray-700 flex items-center px-3">
            <Bot size={16} className="mr-2 text-blue-500" />
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Aetherium Assistant</span>
            {projectContext && (
              <span className="ml-auto text-xs bg-blue-100 dark:bg-blue-900 px-2 py-0.5 rounded">
                {projectContext.project.name}
              </span>
            )}
          </div>

          {/* Chat Messages */}
          <div className="flex-1 overflow-y-auto p-3 space-y-3">
            {aiAgentMessages.map((msg, idx) => (
              <div key={idx} className={`p-3 rounded-lg text-sm ${
                msg.role === 'user'
                  ? 'bg-blue-100 dark:bg-blue-900 ml-4'
                  : 'bg-gray-100 dark:bg-gray-700 mr-4'
              }`}>
                <div>{renderMessageContent(msg.content, idx)}</div>
                <span className="text-xs text-gray-500 mt-1 block">
                  {msg.timestamp.toLocaleTimeString()}
                </span>
              </div>
            ))}
            {isAiThinking && (
              <div className="p-3 rounded-lg bg-gray-100 dark:bg-gray-700 mr-4">
                <div className="flex items-center space-x-2">
                  <RefreshCw size={14} className="animate-spin" />
                  <span className="text-sm">Aetherium is thinking...</span>
                </div>
              </div>
            )}
          </div>

          {/* Chat Input */}
          <div className="p-3 border-t border-gray-300 dark:border-gray-700">
            <div className="flex space-x-2">
              <input
                type="text"
                value={aiAgentInput}
                onChange={(e) => setAiAgentInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && (projectContext ? chatAboutProject() : sendAiMessage())}
                placeholder="Ask Aetherium about your code..."
                className="input flex-1 text-sm"
              />
              <button
                onClick={projectContext ? chatAboutProject : sendAiMessage}
                disabled={isAiThinking}
                className="btn btn-primary text-sm px-3"
              >
                <Bot size={14} />
              </button>
            </div>
          </div>
        </Panel>
      </PanelGroup>

      {/* Bottom Panel */}
      {panelVisible && (
        <PanelGroup direction="vertical" style={{ height: panelCollapsed ? '3vh' : '25vh' }}>
          <Panel defaultSize={100} className="bg-[#252526] border-t border-[#3e3e42] flex flex-col">
            {/* Panel Tabs */}
            <div className="flex bg-[#2d2d30] border-b border-[#3e3e42] items-center">
              {!panelCollapsed && (
                <>
                  <button
                    onClick={() => setPanelView('terminal')}
                    className={`px-4 py-2 text-sm flex items-center text-[#cccccc] ${
                      panelView === 'terminal' ? 'bg-[#1e1e1e] border-t border-[#007acc]' : 'hover:bg-[#37373d]'
                    }`}
                  >
                    <Terminal size={14} className="mr-2" />
                    Terminal
                  </button>
                  <button
                    onClick={() => setPanelView('output')}
                    className={`px-4 py-2 text-sm flex items-center text-[#cccccc] ${
                      panelView === 'output' ? 'bg-[#1e1e1e] border-t border-[#007acc]' : 'hover:bg-[#37373d]'
                    }`}
                  >
                    Output
                  </button>
                  <button
                    onClick={() => setPanelView('problems')}
                    className={`px-4 py-2 text-sm flex items-center text-[#cccccc] ${
                      panelView === 'problems' ? 'bg-[#1e1e1e] border-t border-[#007acc]' : 'hover:bg-[#37373d]'
                    }`}
                  >
                    Problems
                  </button>
                </>
              )}
              <button
                onClick={() => setPanelCollapsed(!panelCollapsed)}
                className="ml-auto p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded text-gray-500 hover:text-gray-300"
                title={panelCollapsed ? "Expand Panel" : "Collapse Panel"}
              >
                <Menu size={14} />
              </button>
            </div>

            {/* Panel Content */}
            {!panelCollapsed && (
              <div className="flex-1 p-3">
                {panelView === 'terminal' && (
                  <div className="h-full bg-black rounded text-green-400 font-mono text-sm">
                    <div className="p-2 overflow-y-auto h-full">
                      <pre className="whitespace-pre-wrap">
                        {terminalOutput || 'Welcome to Aetherium Terminal\n$ '}
                      </pre>
                      <div className="flex items-center mt-2">
                        <span className="text-green-400 mr-2">$</span>
                        <input
                          type="text"
                          value={terminalCommand}
                          onChange={(e) => setTerminalCommand(e.target.value)}
                          onKeyPress={(e) => e.key === 'Enter' && executeCommand()}
                          className="flex-1 bg-transparent border-none outline-none text-green-400"
                          placeholder="Type a command..."
                        />
                      </div>
                    </div>
                  </div>
                )}

                {panelView === 'output' && (
                  <div className="text-sm text-[#cccccc]">
                    Output panel - build results, test output, etc.
                  </div>
                )}

                {panelView === 'problems' && (
                  <div className="text-sm text-[#cccccc]">
                    Problems panel - linting errors, compilation issues, etc.
                  </div>
                )}
              </div>
            )}
          </Panel>
        </PanelGroup>
      )}

      {/* Status Bar */}
      <div className="h-[3vh] bg-[#007acc] border-t border-[#3e3e42] flex items-center px-3 text-xs text-white">
        <div className="flex items-center space-x-4">
          <span>Ready</span>
          {activeTabId && (
            <>
              <span>•</span>
              <span>{openTabs.find(tab => tab.id === activeTabId)?.language || 'plaintext'}</span>
              <span>•</span>
              <span>UTF-8</span>
              <span>•</span>
              <span>LF</span>
            </>
          )}
        </div>
        <div className="ml-auto flex items-center space-x-2">
          <button onClick={() => generateVisualization('dependency')} className="hover:bg-gray-300 dark:hover:bg-gray-600 px-2 py-0.5 rounded">
            <Network size={12} />
          </button>
          <button onClick={() => generateVisualization('call_hierarchy')} className="hover:bg-gray-300 dark:hover:bg-gray-600 px-2 py-0.5 rounded">
            <Eye size={12} />
          </button>
          <button
            onClick={clearWorkspaceState}
            className="hover:bg-gray-300 dark:hover:bg-gray-600 px-2 py-0.5 rounded text-xs"
            title="Clear saved workspace state"
          >
            Clear State
          </button>
          <span>Aetherium: Ready</span>
        </div>
      </div>

      {/* Export Modal */}
      {exportModal.show && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 w-full max-w-md mx-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Export Code to Workspace
            </h3>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Filename
                </label>
                <input
                  type="text"
                  value={exportFilename}
                  onChange={(e) => setExportFilename(e.target.value)}
                  placeholder="Enter filename (e.g., script.py)"
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Path (optional)
                </label>
                <input
                  type="text"
                  value={exportPath}
                  onChange={(e) => setExportPath(e.target.value)}
                  placeholder="Enter path (e.g., src/utils/)"
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                />
              </div>

              <div className="text-sm text-gray-600 dark:text-gray-400">
                Language: {exportModal.language}
              </div>
            </div>

            <div className="flex justify-end space-x-3 mt-6">
              <button
                onClick={closeExportModal}
                className="px-4 py-2 text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200"
              >
                Cancel
              </button>
              <button
                onClick={exportToWorkspace}
                className="px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700"
              >
                Export
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default Workspace