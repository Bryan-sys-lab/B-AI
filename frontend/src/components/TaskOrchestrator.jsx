import { useState, useEffect } from 'react'
import { Plus, Play, GitBranch, CheckCircle, Clock, AlertCircle, RefreshCw } from 'lucide-react'
import { taskAPI } from '../services/api'
import { useNotifications } from '../providers/NotificationProvider'
import { useWebSocket } from '../providers/WebSocketProvider'

function TaskOrchestrator() {
  const [tasks, setTasks] = useState([])
  const [selectedTask, setSelectedTask] = useState(null)
  const [loading, setLoading] = useState(true)
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [newTask, setNewTask] = useState({ description: '', type: 'development' })
  const [taskOutputs, setTaskOutputs] = useState({})
  const { showSuccess, showError } = useNotifications()
  const { messages } = useWebSocket()

  // Helper function to determine language from filename
  const getLanguageFromFilename = (filename) => {
    if (!filename) return 'text'
    const ext = filename.split('.').pop()?.toLowerCase()
    const langMap = {
      'js': 'javascript',
      'jsx': 'javascript',
      'ts': 'typescript',
      'tsx': 'typescript',
      'py': 'python',
      'java': 'java',
      'cpp': 'cpp',
      'c': 'c',
      'go': 'go',
      'rs': 'rust',
      'php': 'php',
      'rb': 'ruby',
      'html': 'html',
      'css': 'css',
      'json': 'json',
      'xml': 'xml',
      'yaml': 'yaml',
      'yml': 'yaml',
      'md': 'markdown',
      'sh': 'shell',
      'sql': 'sql'
    }
    return langMap[ext] || 'text'
  }

  useEffect(() => {
    loadTasks()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Listen for WebSocket messages to update tasks in real-time
  useEffect(() => {
    if (messages.length > 0) {
      const latestMessage = messages[messages.length - 1]
      console.log('Received WebSocket message:', latestMessage)

      if (latestMessage.type === 'task_created') {
        // Reload tasks to include the new one
        loadTasks()
      } else if (latestMessage.type === 'status') {
        // Update task status
        const newStatus = latestMessage.status
        setTasks(prevTasks =>
          prevTasks.map(task =>
            task.id === selectedTask?.id
              ? { ...task, status: newStatus, progress: latestMessage.progress }
              : task
          )
        )
        // Update selected task
        if (selectedTask) {
          setSelectedTask(prev => ({
            ...prev,
            status: newStatus,
            progress: latestMessage.progress
          }))
        }
      } else if (latestMessage.type === 'subtasks') {
        // Update subtasks for the current task
        if (selectedTask) {
          setSelectedTask(prev => ({
            ...prev,
            subtasks: latestMessage.subtasks
          }))
        }
        // Also update in the tasks list
        setTasks(prevTasks =>
          prevTasks.map(task =>
            task.id === selectedTask?.id
              ? { ...task, subtasks: latestMessage.subtasks }
              : task
          )
        )
      } else if (latestMessage.type === 'output') {
        // Store task output for display
        try {
          const outputData = JSON.parse(latestMessage.message)
          console.log('Parsed task output:', outputData)
          setTaskOutputs(prev => {
            const newOutputs = {
              ...prev,
              [latestMessage.task_id]: outputData[0] // Backend sends array with single output
            }
            console.log('Updated taskOutputs:', newOutputs)
            return newOutputs
          })
        } catch (error) {
          console.error('Failed to parse task output:', error, 'Raw message:', latestMessage.message)
          // Store error in taskOutputs
          setTaskOutputs(prev => ({
            ...prev,
            [latestMessage.task_id]: { error: `Failed to parse task output: ${error.message}` }
          }))
        }
      }
    }
  }, [messages, selectedTask]) // eslint-disable-line react-hooks/exhaustive-deps

  const loadTasks = async () => {
    try {
      const response = await taskAPI.list()
      setTasks(response.data)
    } catch (error) {
      showError('Failed to load tasks')
    } finally {
      setLoading(false)
    }
  }

  const createTask = async () => {
    try {
      await taskAPI.create(newTask)
      showSuccess('Task created successfully')
      setNewTask({ description: '', type: 'development' })
      setShowCreateForm(false)
      loadTasks()
    } catch (error) {
      showError('Failed to create task')
    }
  }

  const orchestrateTask = async (taskId) => {
    try {
      await taskAPI.orchestrate(taskId)
      showSuccess('Task orchestration started')
      loadTasks()
    } catch (error) {
      showError('Failed to orchestrate task')
    }
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return 'text-green-600 dark:text-green-400'
      case 'running': return 'text-blue-600 dark:text-blue-400'
      case 'failed': return 'text-red-600 dark:text-red-400'
      case 'pending': return 'text-yellow-600 dark:text-yellow-400'
      default: return 'text-gray-600 dark:text-gray-400'
    }
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed': return <CheckCircle size={16} />
      case 'running': return <RefreshCw size={16} className="animate-spin" />
      case 'failed': return <AlertCircle size={16} />
      case 'pending': return <Clock size={16} />
      default: return <Clock size={16} />
    }
  }

  const renderTaskOutput = (output) => {
    console.log('Rendering task output:', output)
    if (!output) return <p className="text-gray-600 dark:text-gray-400">No output available</p>

    // Handle structured response format from backend
    let processedOutput = output

    // If output has 'structured' field, merge it with the main output
    if (output.structured) {
      processedOutput = {
        ...output,
        explanatory_summary: output.response || output.explanatory_summary,
        file_delivery: output.structured.files ? Object.entries(output.structured.files).map(([filename, content]) => ({
          filename,
          content: typeof content === 'string' ? content : JSON.stringify(content, null, 2),
          language: getLanguageFromFilename(filename)
        })) : output.file_delivery,
        ...output.structured
      }
    }

    return (
      <div className="space-y-4">
        {/* Explanatory Summary */}
        {processedOutput.explanatory_summary && (
          <div>
            <h4 className="font-medium text-gray-900 dark:text-white mb-2">Summary</h4>
            <div className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap">
              {processedOutput.explanatory_summary}
            </div>
          </div>
        )}

        {/* File Delivery */}
        {processedOutput.file_delivery && processedOutput.file_delivery.length > 0 && (
          <div>
            <h4 className="font-medium text-gray-900 dark:text-white mb-2">Generated Code</h4>
            <div className="space-y-2">
              {processedOutput.file_delivery.map((file, index) => (
                <div key={index} className="bg-white dark:bg-gray-600 p-3 rounded border">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-sm">{file.filename}</span>
                    <span className="text-xs text-gray-500 dark:text-gray-400 uppercase">
                      {file.language || 'text'}
                    </span>
                  </div>
                  <pre className="text-xs bg-gray-100 dark:bg-gray-700 p-2 rounded overflow-x-auto">
                    <code>{file.content}</code>
                  </pre>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ZIP Download */}
        {processedOutput.zip_download_url && (
          <div>
            <h4 className="font-medium text-gray-900 dark:text-white mb-2">Download</h4>
            <a
              href={processedOutput.zip_download_url}
              className="inline-flex items-center px-3 py-2 bg-primary-600 text-white text-sm rounded hover:bg-primary-700"
              download={processedOutput.zip_filename || 'project.zip'}
            >
              Download {processedOutput.zip_filename || 'project.zip'}
            </a>
          </div>
        )}

        {/* Inline Code */}
        {processedOutput.inline_code && (
          <div>
            <h4 className="font-medium text-gray-900 dark:text-white mb-2">Code</h4>
            <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded">
              <div dangerouslySetInnerHTML={{ __html: processedOutput.inline_code }} />
            </div>
          </div>
        )}

        {/* Error Display */}
        {processedOutput.error && processedOutput.error.trim() && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded p-3">
            <h4 className="font-medium text-red-800 dark:text-red-400 mb-2">Error</h4>
            <p className="text-sm text-red-700 dark:text-red-300">{processedOutput.error}</p>
          </div>
        )}

        {/* Raw output fallback */}
        {!processedOutput.explanatory_summary && !processedOutput.file_delivery && !processedOutput.zip_download_url && !processedOutput.inline_code && !processedOutput.error && (
          <div>
            <h4 className="font-medium text-gray-900 dark:text-white mb-2">Raw Output</h4>
            <pre className="text-sm bg-gray-100 dark:bg-gray-700 p-3 rounded overflow-x-auto">
              {JSON.stringify(processedOutput, null, 2)}
            </pre>
          </div>
        )}

        {/* Show error if present but not displayed above */}
        {processedOutput.error && !processedOutput.error.trim() && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded p-3">
            <h4 className="font-medium text-red-800 dark:text-red-400 mb-2">Task Failed</h4>
            <p className="text-sm text-red-700 dark:text-red-300">The task completed but returned an error. Please try again.</p>
          </div>
        )}
      </div>
    )
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw size={32} className="animate-spin text-primary-600" />
      </div>
    )
  }

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          Task Orchestrator
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Build and manage complex task workflows with intelligent orchestration
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Task List */}
        <div className="lg:col-span-2">
          <div className="card">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold">Tasks</h2>
              <button
                onClick={() => setShowCreateForm(true)}
                className="btn btn-primary"
              >
                <Plus size={16} className="mr-2" />
                New Task
              </button>
            </div>

            {showCreateForm && (
              <div className="mb-6 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                <h3 className="text-lg font-medium mb-4">Create New Task</h3>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium mb-1">Description</label>
                    <textarea
                      value={newTask.description}
                      onChange={(e) => setNewTask({...newTask, description: e.target.value})}
                      className="input w-full"
                      rows={3}
                      placeholder="Describe the task..."
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">Type</label>
                    <select
                      value={newTask.type}
                      onChange={(e) => setNewTask({...newTask, type: e.target.value})}
                      className="input w-full"
                    >
                      <option value="development">Development</option>
                      <option value="testing">Testing</option>
                      <option value="deployment">Deployment</option>
                      <option value="analysis">Analysis</option>
                      <option value="documentation">Documentation</option>
                    </select>
                  </div>
                  <div className="flex space-x-2">
                    <button onClick={createTask} className="btn btn-primary">
                      Create Task
                    </button>
                    <button
                      onClick={() => setShowCreateForm(false)}
                      className="btn btn-secondary"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              </div>
            )}

            <div className="space-y-4">
              {tasks.map((task) => (
                <div
                  key={task.id}
                  className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                    selectedTask?.id === task.id
                      ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                  }`}
                  onClick={() => setSelectedTask(task)}
                >
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-medium text-gray-900 dark:text-white">
                      {task.description}
                    </h3>
                    <div className={`flex items-center ${getStatusColor(task.status)}`}>
                      {getStatusIcon(task.status)}
                      <span className="ml-1 text-sm capitalize">{task.status}</span>
                    </div>
                  </div>

                  <div className="flex items-center justify-between text-sm text-gray-600 dark:text-gray-400">
                    <span>Type: {task.type}</span>
                    <span>Created: {new Date(task.created_at).toLocaleDateString()}</span>
                  </div>

                  {task.subtasks && task.subtasks.length > 0 && (
                    <div className="mt-2 flex items-center text-sm text-gray-600 dark:text-gray-400">
                      <GitBranch size={14} className="mr-1" />
                      {task.subtasks.length} subtasks
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Task Details */}
        <div className="lg:col-span-1">
          {selectedTask ? (
            <div className="card">
              <h2 className="text-xl font-semibold mb-4">Task Details</h2>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                    Status
                  </label>
                  <div className={`flex items-center ${getStatusColor(selectedTask.status)}`}>
                    {getStatusIcon(selectedTask.status)}
                    <span className="ml-2 capitalize">{selectedTask.status}</span>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                    Type
                  </label>
                  <span className="capitalize">{selectedTask.type}</span>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                    Created
                  </label>
                  <span>{new Date(selectedTask.created_at).toLocaleString()}</span>
                </div>

                {selectedTask.progress && (
                  <div>
                    <label className="block text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                      Progress
                    </label>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div
                        className="bg-primary-600 h-2 rounded-full"
                        style={{ width: `${selectedTask.progress}%` }}
                      />
                    </div>
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      {selectedTask.progress}%
                    </span>
                  </div>
                )}

                <div className="pt-4">
                  <button
                    onClick={() => orchestrateTask(selectedTask.id)}
                    disabled={selectedTask.status === 'running'}
                    className="btn btn-primary w-full disabled:opacity-50"
                  >
                    <Play size={16} className="mr-2" />
                    {selectedTask.status === 'running' ? 'Running...' : 'Orchestrate Task'}
                  </button>
                </div>
              </div>

              {/* Subtasks */}
              {selectedTask.subtasks && selectedTask.subtasks.length > 0 && (
                <div className="mt-6">
                  <h3 className="text-lg font-medium mb-3">Subtasks</h3>
                  <div className="space-y-2">
                    {selectedTask.subtasks.map((subtask, index) => (
                      <div key={index} className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-700 rounded">
                        <span className="text-sm">{subtask.description}</span>
                        <div className={`flex items-center ${getStatusColor(subtask.status)}`}>
                          {getStatusIcon(subtask.status)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Task Output */}
              {taskOutputs[selectedTask.id] && (
                <div className="mt-6">
                  <h3 className="text-lg font-medium mb-3">Results</h3>
                  <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                    {renderTaskOutput(taskOutputs[selectedTask.id])}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="card">
              <div className="text-center text-gray-500 dark:text-gray-400 py-12">
                <GitBranch size={48} className="mx-auto mb-4 opacity-50" />
                <p>Select a task to view details</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default TaskOrchestrator