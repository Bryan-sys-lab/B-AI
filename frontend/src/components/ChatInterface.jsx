import { useState, useEffect, useRef } from 'react'
import { Send, Bot, User, Loader2, Trash2, Copy, FileText, Play, Terminal, Download, X, History, Plus, Calendar } from 'lucide-react'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { taskAPI, systemAPI, workspaceAPI } from '../services/api'
import { useWebSocket } from '../providers/WebSocketProvider'
import { useNotifications } from '../providers/NotificationProvider'
import { logUserAction, logError } from '../services/logger.js'

function ChatInterface() {
  const [messages, setMessages] = useState(() => {
    // Load messages from localStorage on initial render
    try {
      const saved = localStorage.getItem('chatMessages')
      return saved ? JSON.parse(saved) : []
    } catch (e) {
      return []
    }
  })
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [taskId, setTaskId] = useState(null)
  const [codeExecutionResults, setCodeExecutionResults] = useState({})
  const [runningCodeBlocks, setRunningCodeBlocks] = useState(new Set())
  const [exportModal, setExportModal] = useState({ show: false, code: '', language: '', messageIndex: -1, codeBlockIndex: -1 })
  const [exportFilename, setExportFilename] = useState('')
  const [exportPath, setExportPath] = useState('')
  const [showTerminal, setShowTerminal] = useState(false)
  const [terminalOutput, setTerminalOutput] = useState('')
  const [terminalCommand, setTerminalCommand] = useState('')
  const [showHistory, setShowHistory] = useState(false)
  const [savedChats, setSavedChats] = useState(() => {
    try {
      const saved = localStorage.getItem('savedChats')
      return saved ? JSON.parse(saved) : []
    } catch (e) {
      return []
    }
  })
  const [currentChatId, setCurrentChatId] = useState(null)
  const [compressedBlocks, setCompressedBlocks] = useState(new Set())
  const [compressedMessages, setCompressedMessages] = useState(new Set())
  const [showScrollToBottom, setShowScrollToBottom] = useState(false)
  const [showScrollToTop, setShowScrollToTop] = useState(false)
  const messagesEndRef = useRef(null)
  const messagesContainerRef = useRef(null)
  const { sendMessage, messages: wsMessages } = useWebSocket()
  const { showSuccess, showError } = useNotifications()

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const scrollToTop = () => {
    messagesContainerRef.current?.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const toggleCompression = (blockId) => {
    setCompressedBlocks(prev => {
      const newSet = new Set(prev)
      if (newSet.has(blockId)) {
        newSet.delete(blockId)
      } else {
        newSet.add(blockId)
      }
      return newSet
    })
  }

  const toggleMessageCompression = (messageIndex) => {
    setCompressedMessages(prev => {
      const newSet = new Set(prev)
      if (newSet.has(messageIndex)) {
        newSet.delete(messageIndex)
      } else {
        newSet.add(messageIndex)
      }
      return newSet
    })
  }

  const handleScroll = () => {
    if (messagesContainerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = messagesContainerRef.current
      const isNearBottom = scrollTop + clientHeight >= scrollHeight - 100
      const isNearTop = scrollTop <= 100

      setShowScrollToBottom(!isNearBottom)
      setShowScrollToTop(!isNearTop)
    }
  }

  const clearChat = () => {
    setMessages([])
    setCurrentChatId(null)
    try {
      localStorage.removeItem('chatMessages')
    } catch (e) {
      console.warn('Failed to clear chat messages from localStorage:', e)
    }
  }

  const saveCurrentChat = () => {
    if (messages.length === 0) return

    const chatTitle = messages[0]?.content?.slice(0, 50) || 'New Chat'
    const chatId = currentChatId || Date.now().toString()
    const chatData = {
      id: chatId,
      title: chatTitle,
      messages: messages,
      timestamp: new Date().toISOString()
    }

    const updatedChats = savedChats.filter(chat => chat.id !== chatId)
    updatedChats.unshift(chatData) // Add to beginning

    // Keep only last 50 chats
    if (updatedChats.length > 50) {
      updatedChats.splice(50)
    }

    setSavedChats(updatedChats)
    setCurrentChatId(chatId)

    try {
      localStorage.setItem('savedChats', JSON.stringify(updatedChats))
      showSuccess('Chat saved to history')
    } catch (e) {
      console.warn('Failed to save chat to history:', e)
      showError('Failed to save chat')
    }
  }

  const loadChat = (chatId) => {
    const chat = savedChats.find(c => c.id === chatId)
    if (chat) {
      setMessages(chat.messages)
      setCurrentChatId(chat.id)
      setShowHistory(false)
      showSuccess('Chat loaded')
    }
  }

  const deleteChat = (chatId) => {
    const updatedChats = savedChats.filter(chat => chat.id !== chatId)
    setSavedChats(updatedChats)

    try {
      localStorage.setItem('savedChats', JSON.stringify(updatedChats))
      showSuccess('Chat deleted')
    } catch (e) {
      console.warn('Failed to delete chat from history:', e)
      showError('Failed to delete chat')
    }
  }

  const startNewChat = () => {
    if (messages.length > 0) {
      saveCurrentChat()
    }
    clearChat()
  }

  // Copy functionality with fallback for older browsers/HTTP
  const copyToClipboard = async (text) => {
    try {
      // Try modern Clipboard API first (requires HTTPS)
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(text)
        showSuccess('Copied to clipboard')
        return
      }
    } catch (err) {
      console.warn('Modern clipboard API failed, trying fallback:', err)
    }

    // Fallback for older browsers or HTTP contexts
    try {
      const textArea = document.createElement('textarea')
      textArea.value = text
      textArea.style.position = 'fixed'
      textArea.style.left = '-999999px'
      textArea.style.top = '-999999px'
      document.body.appendChild(textArea)
      textArea.focus()
      textArea.select()

      const successful = document.execCommand('copy')
      document.body.removeChild(textArea)

      if (successful) {
        showSuccess('Copied to clipboard')
      } else {
        throw new Error('Fallback copy method failed')
      }
    } catch (fallbackErr) {
      console.error('All copy methods failed:', fallbackErr)
      showError('Failed to copy to clipboard. Please try selecting and copying manually.')
    }
  }

  const copyMessage = (message) => {
    copyToClipboard(message.content)
  }

  const copyConversation = () => {
    const conversationText = messages
      .map(msg => `${msg.role === 'user' ? 'User' : 'Assistant'}: ${msg.content}`)
      .join('\n\n')
    copyToClipboard(conversationText)
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

  const executeTerminalCommand = async () => {
    if (!terminalCommand.trim()) return

    try {
      const response = await systemAPI.executeShell({ command: terminalCommand })
      setTerminalOutput(prev => prev + `\n$ ${terminalCommand}\n${response.data.output}`)
      setTerminalCommand('')
    } catch (error) {
      setTerminalOutput(prev => prev + `\n$ ${terminalCommand}\nError: ${error.response?.data?.detail || 'Command failed'}`)
    }
  }

  const toggleTerminal = () => {
    setShowTerminal(!showTerminal)
    if (!showTerminal) {
      // Initialize terminal with welcome message
      setTerminalOutput('Welcome to Aetherium Terminal\nType commands below...\n')
    }
  }

  // Parse message content to separate text and code blocks
  const parseMessageContent = (content, isMessageCompressed = false) => {
    if (isMessageCompressed) {
      // Show compressed version of the entire message
      const previewLength = 200
      const preview = content.length > previewLength
        ? content.substring(0, previewLength) + '... (click ⤢ to expand)'
        : content
      return [{ type: 'text', content: preview }]
    }

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
  const renderMessageContent = (content, messageIndex, isMessageCompressed = false) => {
    const parts = parseMessageContent(content, isMessageCompressed)

    return parts.map((part, partIndex) => {
      if (part.type === 'code') {
        const codeBlockId = `${messageIndex}-${partIndex}`
        const executionResult = codeExecutionResults[codeBlockId]
        const isRunning = runningCodeBlocks.has(codeBlockId)
        const canRun = ['python', 'py', 'javascript', 'js', 'node'].includes(part.language.toLowerCase())
        const isCompressed = compressedBlocks.has(codeBlockId)
        const isLongCode = part.content.split('\n').length > 20

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
                borderRadius: executionResult ? '0' : (isLongCode ? '0' : '0 0 6px 6px'),
                fontSize: '14px',
                maxHeight: isCompressed ? '200px' : 'none',
                overflow: isCompressed ? 'hidden' : 'visible'
              }}
            >
              {isCompressed ? part.content.split('\n').slice(0, 10).join('\n') + '\n... (truncated)' : part.content}
            </SyntaxHighlighter>
            {isLongCode && (
              <div className="flex justify-center border-t border-gray-700 bg-gray-800 dark:bg-gray-900 rounded-b-md">
                <button
                  onClick={() => toggleCompression(codeBlockId)}
                  className="text-gray-400 hover:text-white transition-colors py-2 px-4 text-sm"
                  title={isCompressed ? "Expand code" : "Compress code"}
                >
                  {isCompressed ? '⤢ Expand Code' : '⤡ Compress Code'}
                </button>
              </div>
            )}
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
        // Handle long text sections with compression
        const textBlockId = `text-${messageIndex}-${partIndex}`
        const isCompressed = compressedBlocks.has(textBlockId)
        const isLongText = part.content.length > 500
        const displayText = isCompressed ? part.content.substring(0, 300) + '... (click ⤢ to expand)' : part.content

        return (
          <div key={partIndex} className="relative">
            <span className="whitespace-pre-wrap">{displayText}</span>
            {isLongText && (
              <button
                onClick={() => toggleCompression(textBlockId)}
                className="ml-2 text-xs text-blue-400 hover:text-blue-300 transition-colors"
                title={isCompressed ? "Expand text" : "Compress text"}
              >
                {isCompressed ? '⤢ Expand' : '⤡ Compress'}
              </button>
            )}
          </div>
        )
      }
    })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Save messages to localStorage whenever they change
  useEffect(() => {
    try {
      localStorage.setItem('chatMessages', JSON.stringify(messages))
    } catch (e) {
      console.warn('Failed to save chat messages to localStorage:', e)
    }
  }, [messages])

  // Handle scroll detection for scroll-to-bottom button
  useEffect(() => {
    const container = messagesContainerRef.current
    if (container) {
      container.addEventListener('scroll', handleScroll)
      return () => container.removeEventListener('scroll', handleScroll)
    }
  }, [])

  // Handle WebSocket messages
  useEffect(() => {
    if (wsMessages.length > 0) {
      const latestMessage = wsMessages[wsMessages.length - 1]
      console.log('Processing WebSocket message:', latestMessage)

      if (latestMessage.type === 'task_created') {
        console.log('Received task_created message:', latestMessage)
        // If we don't have a taskId yet (API call failed), set it from WebSocket
        if (!taskId && latestMessage.task_id) {
          console.log('Setting taskId from WebSocket task_created:', latestMessage.task_id)
          setTaskId(latestMessage.task_id)
          // Update the current Aetherium message with the taskId
          setMessages(prev => prev.map(msg =>
            msg.isStreaming && !msg.taskId ? { ...msg, taskId: latestMessage.task_id } : msg
          ))
        }
      } else if (latestMessage.type === 'output' && (latestMessage.task_id || taskId)) {
        const messageTaskId = latestMessage.task_id || taskId
        console.log('Processing output message for task:', messageTaskId)
        console.log('Raw WebSocket message:', latestMessage)

        // Update the Aetherium message with the actual response
        setMessages(prev => prev.map(msg => {
          console.log('Checking message with taskId:', msg.taskId, 'against', messageTaskId)
          if (msg.taskId === messageTaskId && msg.isStreaming) {
            console.log('Found matching streaming message, updating content')

            // Use the raw message content directly - let the backend format it
            let content = latestMessage.message || ''
            let structuredOutput = null

            // Try to parse as JSON for structured processing, but fall back to raw content
            try {
              const outputData = JSON.parse(latestMessage.message)
              console.log('Parsed output data:', outputData)

              if (Array.isArray(outputData) && outputData.length > 0) {
                structuredOutput = outputData[0]
              } else {
                structuredOutput = outputData
              }

              // If it's structured data, try to extract meaningful content
              if (structuredOutput && typeof structuredOutput === 'object') {
                if (structuredOutput.error) {
                  content = structuredOutput.error
                  console.log('Found error in output:', structuredOutput.error)
                } else if (structuredOutput.explanatory_summary) {
                  content = structuredOutput.explanatory_summary
                  console.log('Using explanatory_summary')

                  // Add file delivery if present
                  if (structuredOutput.file_delivery && structuredOutput.file_delivery.length > 0) {
                    const additionalFiles = structuredOutput.file_delivery.filter(file =>
                      file.language !== 'text' && !content.includes(file.filename)
                    )
                    if (additionalFiles.length > 0) {
                      content += '\n\nGenerated Files:\n'
                      additionalFiles.forEach((file, index) => {
                        content += `${file.filename} (${file.language}):\n\`\`\`${file.language}\n${file.content}\n\`\`\`\n`
                      })
                    }
                  }
                } else if (structuredOutput.response) {
                  content = structuredOutput.response
                  console.log('Using response field')
                } else {
                  // For other structured data, keep as raw JSON
                  content = JSON.stringify(structuredOutput, null, 2)
                }
              }
            } catch (e) {
              // If JSON parsing fails, use the raw message content
              console.log('JSON parsing failed, using raw message content')
              content = latestMessage.message
            }

            console.log('Final content preview:', content.substring(0, 200) + '...')

            // Append explanation and run steps if present (especially important when files are generated)
            if (latestMessage.explanation) {
              // Clean up only # and -- markers from explanation while preserving other markdown
              let cleanExplanation = latestMessage.explanation
                .replace(/^#+\s*/gm, '') // Remove heading markers (#)
                .replace(/^--+\s*/gm, '') // Remove dash markers (--)
                .trim()
              content += '\n\n' + cleanExplanation
              console.log('Added cleaned explanation to content')
            }

            if (latestMessage.run_steps && latestMessage.run_steps.length > 0) {
              content += '\n\n**Run Steps:**\n' + latestMessage.run_steps.map((step, index) => `${index + 1}. \`${step}\``).join('\n')
              console.log('Added run steps to content')
            }

            return {
              ...msg,
              content,
              structuredOutput,
              isStreaming: false
            }
          }
          return msg
        }))

        // Stop loading when we get a response
        setIsLoading(false)
      } else if (latestMessage.type === 'status' && latestMessage.status === 'completed') {
        console.log('Received completed status, stopping loading')
        // Ensure loading is stopped
        setIsLoading(false)
      }
    }
  }, [wsMessages, taskId])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const messageContent = input.trim()
    logUserAction('chat_message_sent', {
      messageLength: messageContent.length,
      wordCount: messageContent.split(' ').length
    })

    const userMessage = { role: 'user', content: messageContent, timestamp: new Date() }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    try {
      // Prepare conversation context for better follow-up handling
      const conversationHistory = messages.slice(-10).map(msg => ({
        role: msg.role,
        content: msg.content,
        timestamp: msg.timestamp
      }))

      const response = await taskAPI.create({
        description: messageContent,
        type: 'chat',
        context: {
          conversation_history: conversationHistory
        }
      })

      console.log('Task creation response:', response)
      console.log('Response data:', response.data)
      console.log('Response data keys:', Object.keys(response.data || {}))
      const extractedTaskId = response.data?.task_id || response.data?.id
      console.log('Extracted taskId:', extractedTaskId)
      setTaskId(extractedTaskId)
      logUserAction('task_created', { taskId: extractedTaskId, description: messageContent })

      // Check if response contains direct response data
      if (response.data?.response && response.data?.type) {
        console.log('Direct response received, displaying immediately')
        // Add Aetherium response directly
        const aiMessage = {
          role: 'assistant',
          content: response.data.response,
          timestamp: new Date(),
          taskId: extractedTaskId,
          isStreaming: false
        }
        setMessages(prev => [...prev, aiMessage])
        setIsLoading(false)
        return
      }

      // Add initial Aetherium response placeholder for async responses
      const aiMessage = {
        role: 'assistant',
        content: '',
        timestamp: new Date(),
        taskId: extractedTaskId,
        isStreaming: true
      }
      setMessages(prev => {
        const newMessages = [...prev, aiMessage]
        // Auto-save chat when starting a conversation
        if (newMessages.length === 2) { // User message + AI placeholder
          setTimeout(() => saveCurrentChat(), 100) // Small delay to ensure state is updated
        }
        return newMessages
      })

      // Listen for WebSocket updates
      console.log('Sending subscribe_task message with taskId:', extractedTaskId)
      if (extractedTaskId) {
        sendMessage({ type: 'subscribe_task', taskId: extractedTaskId })
      } else {
        console.error('No taskId available for subscription')
      }

    } catch (error) {
      logError(error, { context: 'chat_message_send', message: messageContent })
      console.log('API call failed, but task may still be processing via WebSocket')

      // Even if API call fails, create Aetherium message placeholder and wait for WebSocket
      const aiMessage = {
        role: 'assistant',
        content: '',
        timestamp: new Date(),
        taskId: null, // Will be set by WebSocket task_created message
        isStreaming: true
      }
      setMessages(prev => [...prev, aiMessage])

      // Don't show error immediately - task might still be processing
      // showError('Failed to send message')
    }
  }

  return (
    <div className="h-screen flex flex-col max-w-7xl mx-auto relative overflow-hidden">
      {/* Header - Fixed at top */}
      <div className="fixed top-16 left-0 right-0 z-20 bg-white dark:bg-gray-900 p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              Aetherium Chat
            </h1>
            <p className="text-sm text-gray-600 dark:text-gray-400 hidden sm:block">
              Interact with Aetherium agents for development assistance
            </p>
          </div>
          <div className="flex space-x-1 sm:space-x-2">
            <button
              onClick={() => setShowHistory(true)}
              className="btn btn-secondary p-2 sm:px-4 sm:py-2 flex items-center space-x-1 sm:space-x-2"
              title="View chat history"
            >
              <History size={16} />
              <span className="hidden sm:inline">History</span>
            </button>
            <button
              onClick={startNewChat}
              className="btn btn-secondary p-2 sm:px-4 sm:py-2 flex items-center space-x-1 sm:space-x-2"
              title="Start new conversation"
            >
              <Plus size={16} />
              <span className="hidden sm:inline">New Chat</span>
            </button>
            {messages.length > 0 && (
              <button
                onClick={clearChat}
                className="btn btn-secondary p-2 sm:px-4 sm:py-2 flex items-center space-x-1 sm:space-x-2 text-red-600 hover:text-red-700"
                title="Delete current chat"
              >
                <Trash2 size={16} />
                <span className="hidden sm:inline">Delete</span>
              </button>
            )}
            <button
              onClick={toggleTerminal}
              className={`btn ${showTerminal ? 'btn-primary' : 'btn-secondary'} p-2 sm:px-4 sm:py-2 flex items-center space-x-1 sm:space-x-2`}
              title={showTerminal ? 'Hide terminal' : 'Show terminal'}
            >
              <Terminal size={16} />
              <span className="hidden sm:inline">Terminal</span>
            </button>
            {messages.length > 0 && (
              <button
                onClick={copyConversation}
                className="btn btn-secondary p-2 sm:px-4 sm:py-2 flex items-center space-x-1 sm:space-x-2"
                title="Copy entire conversation"
              >
                <FileText size={16} />
                <span className="hidden sm:inline">Copy Chat</span>
              </button>
            )}
          </div>
          </div>
        </div>
      </div>

      {/* Chat Messages - Scrollable area */}
      <div
        ref={messagesContainerRef}
        className="flex-1 bg-white dark:bg-gray-800 overflow-y-auto pt-24 pb-40 px-4"
      >
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-500 dark:text-gray-400">
            <div className="text-center">
              <Bot size={48} className="mx-auto mb-4 opacity-50" />
              <p className="text-lg">Start a conversation with Aetherium</p>
              <p className="text-sm">Ask questions, request code help, or get development assistance</p>
            </div>
          </div>
        ) : (
          <div className="space-y-4 pb-4">
            {messages.map((message, index) => (
              message.role === 'user' ? (
                // User message with bubble
                <div key={index} className="flex items-start space-x-3 justify-end">
                  <div className="max-w-xs lg:max-w-md px-4 py-2 rounded-lg relative group bg-primary-600 text-white">
                    <div>
                      {renderMessageContent(message.content, index)}
                      {/* Copy message button */}
                      <button
                        onClick={() => copyMessage(message)}
                        className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity text-white hover:text-gray-200"
                        title="Copy message"
                      >
                        <Copy size={14} />
                      </button>
                    </div>
                  </div>
                  <div className="flex-shrink-0 w-8 h-8 bg-gray-300 dark:bg-gray-600 rounded-full flex items-center justify-center">
                    <User size={16} className="text-gray-600 dark:text-gray-300" />
                  </div>
                </div>
              ) : (
                // Aetherium message without bubble
                <div key={index} className="flex items-start space-x-3">
                  <div className="flex-shrink-0 w-8 h-8 bg-primary-100 dark:bg-primary-900 rounded-full flex items-center justify-center">
                    <Bot size={16} className="text-primary-600 dark:text-primary-400" />
                  </div>
                  <div className="flex-1 max-w-4xl">
                    {message.isStreaming ? (
                      <div className="flex items-center space-x-2 text-gray-600 dark:text-gray-400">
                        <Loader2 size={16} className="animate-spin" />
                        <span>Aetherium is thinking...</span>
                      </div>
                    ) : (
                      <div className="relative group">
                        {renderMessageContent(message.content, index, compressedMessages.has(index))}
                        {/* Message compression button */}
                        {message.content && message.content.length > 1000 && (
                          <button
                            onClick={() => toggleMessageCompression(index)}
                            className="absolute top-2 right-8 opacity-0 group-hover:opacity-100 transition-opacity text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 bg-white dark:bg-gray-800 rounded p-1 shadow-sm"
                            title={compressedMessages.has(index) ? "Expand message" : "Compress message"}
                          >
                            {compressedMessages.has(index) ? '⤢' : '⤡'}
                          </button>
                        )}
                        {/* Copy message button */}
                        <button
                          onClick={() => copyMessage(message)}
                          className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 bg-white dark:bg-gray-800 rounded p-1 shadow-sm"
                          title="Copy message"
                        >
                          <Copy size={14} />
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              )
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}

        {/* Scroll Buttons */}
        {showScrollToTop && (
          <button
            onClick={scrollToTop}
            className="fixed bottom-48 right-6 bg-primary-600 hover:bg-primary-700 text-white p-3 rounded-full shadow-lg transition-all duration-200 z-30"
            title="Scroll to top"
          >
            ↑
          </button>
        )}
        {showScrollToBottom && (
          <button
            onClick={scrollToBottom}
            className="fixed bottom-40 right-6 bg-primary-600 hover:bg-primary-700 text-white p-3 rounded-full shadow-lg transition-all duration-200 z-30"
            title="Scroll to bottom"
          >
            ↓
          </button>
        )}
      </div>

      {/* Terminal - Fixed height when shown */}
      {showTerminal && (
        <div className="flex-shrink-0 bg-gray-900 border-t border-gray-700 p-4 h-48">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center">
              <Terminal size={18} className="text-green-400 mr-2" />
              <h3 className="text-sm font-semibold text-white">Terminal</h3>
            </div>
            <button
              onClick={toggleTerminal}
              className="text-gray-400 hover:text-white"
              title="Close terminal"
            >
              <X size={16} />
            </button>
          </div>

          <div className="bg-black rounded p-3 h-24 overflow-y-auto mb-3 font-mono text-xs text-green-400">
            <pre className="whitespace-pre-wrap">{terminalOutput}</pre>
          </div>

          <div className="flex space-x-2">
            <input
              type="text"
              value={terminalCommand}
              onChange={(e) => setTerminalCommand(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && executeTerminalCommand()}
              placeholder="Enter command..."
              className="input flex-1 text-sm bg-gray-800 border-gray-600 text-white placeholder-gray-400 h-8"
            />
            <button onClick={executeTerminalCommand} className="btn btn-primary px-3 py-1 h-8">
              <Play size={14} />
            </button>
          </div>
        </div>
      )}

      {/* Input Form - Above footer */}
      <div className="fixed bottom-32 left-0 right-0 z-20 p-4 border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 max-w-7xl mx-auto">
        <form onSubmit={handleSubmit} className="flex space-x-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            className="input flex-1"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="btn btn-primary px-4 py-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <Loader2 size={18} className="animate-spin" />
            ) : (
              <Send size={18} />
            )}
          </button>
        </form>
      </div>

      {/* History Modal */}
      {showHistory && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 w-full max-w-2xl mx-4 max-h-[80vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                Chat History
              </h3>
              <button
                onClick={() => setShowHistory(false)}
                className="text-gray-400 hover:text-white"
                title="Close history"
              >
                <X size={20} />
              </button>
            </div>

            {savedChats.length === 0 ? (
              <p className="text-gray-600 dark:text-gray-400 text-center py-8">
                No saved chats yet. Start a conversation and it will be saved automatically.
              </p>
            ) : (
              <div className="space-y-2">
                {savedChats.map((chat) => (
                  <div key={chat.id} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                    <div className="flex-1 min-w-0">
                      <h4 className="text-sm font-medium text-gray-900 dark:text-white truncate">
                        {chat.title}
                      </h4>
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        {new Date(chat.timestamp).toLocaleString()}
                      </p>
                    </div>
                    <div className="flex space-x-2">
                      <button
                        onClick={() => loadChat(chat.id)}
                        className="px-3 py-1 text-xs bg-primary-600 text-white rounded hover:bg-primary-700"
                      >
                        Load
                      </button>
                      <button
                        onClick={() => deleteChat(chat.id)}
                        className="px-3 py-1 text-xs bg-red-600 text-white rounded hover:bg-red-700"
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}

            <div className="flex justify-end mt-6">
              <button
                onClick={() => setShowHistory(false)}
                className="px-4 py-2 text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

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

export default ChatInterface