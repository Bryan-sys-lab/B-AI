import { useState, useEffect } from 'react'
import { FileText, Plus, Edit, Trash2, Play, Copy, Star, RefreshCw } from 'lucide-react'
import { promptAPI } from '../services/api'
import { useNotifications } from '../providers/NotificationProvider'

function PromptEngineeringStudio() {
  const [prompts, setPrompts] = useState([])
  const [loading, setLoading] = useState(true)
  const [selectedPrompt, setSelectedPrompt] = useState(null)
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [editingPrompt, setEditingPrompt] = useState(null)
  const [formData, setFormData] = useState({
    title: '',
    content: '',
    category: 'general',
    tags: '',
    is_favorite: false
  })
  const [testInput, setTestInput] = useState('')
  const [testOutput, setTestOutput] = useState('')
  const [selectedCategory, setSelectedCategory] = useState('all')
  const { showSuccess, showError } = useNotifications()

  useEffect(() => {
    loadPrompts()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const loadPrompts = async () => {
    try {
      const response = await promptAPI.list()
      setPrompts(response.data)
    } catch (error) {
      showError('Failed to load prompts')
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    try {
      const promptData = {
        ...formData,
        tags: formData.tags.split(',').map(tag => tag.trim()).filter(tag => tag)
      }

      if (editingPrompt) {
        await promptAPI.update(editingPrompt.id, promptData)
        showSuccess('Prompt updated successfully')
      } else {
        await promptAPI.create(promptData)
        showSuccess('Prompt created successfully')
      }
      resetForm()
      loadPrompts()
    } catch (error) {
      showError(`Failed to ${editingPrompt ? 'update' : 'create'} prompt`)
    }
  }

  const handleEdit = (prompt) => {
    setEditingPrompt(prompt)
    setFormData({
      title: prompt.title,
      content: prompt.content,
      category: prompt.category,
      tags: prompt.tags ? prompt.tags.join(', ') : '',
      is_favorite: prompt.is_favorite
    })
    setShowCreateForm(true)
  }

  const handleDelete = async (promptId) => {
    if (!confirm('Are you sure you want to delete this prompt?')) return

    try {
      await promptAPI.delete(promptId)
      showSuccess('Prompt deleted successfully')
      loadPrompts()
      if (selectedPrompt?.id === promptId) {
        setSelectedPrompt(null)
      }
    } catch (error) {
      showError('Failed to delete prompt')
    }
  }

  const toggleFavorite = async (prompt) => {
    try {
      await promptAPI.update(prompt.id, { is_favorite: !prompt.is_favorite })
      loadPrompts()
    } catch (error) {
      showError('Failed to update favorite status')
    }
  }

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text)
    showSuccess('Copied to clipboard')
  }

  const resetForm = () => {
    setFormData({
      title: '',
      content: '',
      category: 'general',
      tags: '',
      is_favorite: false
    })
    setEditingPrompt(null)
    setShowCreateForm(false)
  }

  const filteredPrompts = selectedCategory === 'all'
    ? prompts
    : prompts.filter(prompt => prompt.category === selectedCategory)

  const categories = ['all', ...new Set(prompts.map(p => p.category))]

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
          Prompt Engineering Studio
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Create, test, and manage Aetherium prompts for optimal performance
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Prompts List */}
        <div className="lg:col-span-1">
          <div className="card">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold">Prompts</h2>
              <button
                onClick={() => setShowCreateForm(true)}
                className="btn btn-primary text-sm"
              >
                <Plus size={16} className="mr-1" />
                New
              </button>
            </div>

            {/* Category Filter */}
            <div className="mb-4">
              <select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                className="input w-full text-sm"
              >
                {categories.map(category => (
                  <option key={category} value={category}>
                    {category === 'all' ? 'All Categories' : category.charAt(0).toUpperCase() + category.slice(1)}
                  </option>
                ))}
              </select>
            </div>

            {/* Create/Edit Form */}
            {showCreateForm && (
              <div className="mb-4 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                <h3 className="text-sm font-medium mb-3">
                  {editingPrompt ? 'Edit Prompt' : 'Create New Prompt'}
                </h3>

                <form onSubmit={handleSubmit} className="space-y-3">
                  <input
                    type="text"
                    placeholder="Prompt title"
                    value={formData.title}
                    onChange={(e) => setFormData({...formData, title: e.target.value})}
                    className="input w-full text-sm"
                    required
                  />

                  <select
                    value={formData.category}
                    onChange={(e) => setFormData({...formData, category: e.target.value})}
                    className="input w-full text-sm"
                  >
                    <option value="general">General</option>
                    <option value="coding">Coding</option>
                    <option value="writing">Writing</option>
                    <option value="analysis">Analysis</option>
                    <option value="creative">Creative</option>
                  </select>

                  <textarea
                    placeholder="Prompt content..."
                    value={formData.content}
                    onChange={(e) => setFormData({...formData, content: e.target.value})}
                    className="input w-full text-sm"
                    rows={4}
                    required
                  />

                  <input
                    type="text"
                    placeholder="Tags (comma separated)"
                    value={formData.tags}
                    onChange={(e) => setFormData({...formData, tags: e.target.value})}
                    className="input w-full text-sm"
                  />

                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="is_favorite"
                      checked={formData.is_favorite}
                      onChange={(e) => setFormData({...formData, is_favorite: e.target.checked})}
                    />
                    <label htmlFor="is_favorite" className="text-sm">Mark as favorite</label>
                  </div>

                  <div className="flex space-x-2">
                    <button type="submit" className="btn btn-primary text-sm flex-1">
                      {editingPrompt ? 'Update' : 'Create'}
                    </button>
                    <button type="button" onClick={resetForm} className="btn btn-secondary text-sm">
                      Cancel
                    </button>
                  </div>
                </form>
              </div>
            )}

            {/* Prompts List */}
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {filteredPrompts.map((prompt) => (
                <div
                  key={prompt.id}
                  className={`p-3 border rounded-lg cursor-pointer transition-colors ${
                    selectedPrompt?.id === prompt.id
                      ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                  }`}
                  onClick={() => setSelectedPrompt(prompt)}
                >
                  <div className="flex items-start justify-between mb-1">
                    <h3 className="font-medium text-sm text-gray-900 dark:text-white line-clamp-1">
                      {prompt.title}
                    </h3>
                    <div className="flex items-center space-x-1">
                      {prompt.is_favorite && (
                        <Star size={14} className="text-yellow-500 fill-current" />
                      )}
                      <span className={`px-1.5 py-0.5 text-xs rounded ${getCategoryColor(prompt.category)}`}>
                        {prompt.category}
                      </span>
                    </div>
                  </div>

                  <p className="text-xs text-gray-600 dark:text-gray-400 line-clamp-2 mb-2">
                    {prompt.content.substring(0, 100)}...
                  </p>

                  {prompt.tags && prompt.tags.length > 0 && (
                    <div className="flex flex-wrap gap-1">
                      {prompt.tags.slice(0, 2).map((tag, index) => (
                        <span key={index} className="px-1.5 py-0.5 text-xs bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 rounded">
                          {tag}
                        </span>
                      ))}
                      {prompt.tags.length > 2 && (
                        <span className="px-1.5 py-0.5 text-xs text-gray-500">
                          +{prompt.tags.length - 2}
                        </span>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>

            {filteredPrompts.length === 0 && (
              <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                <FileText size={32} className="mx-auto mb-2 opacity-50" />
                <p className="text-sm">No prompts found</p>
              </div>
            )}
          </div>
        </div>

        {/* Prompt Editor */}
        <div className="lg:col-span-2">
          {selectedPrompt ? (
            <div className="space-y-6">
              {/* Prompt Header */}
              <div className="card">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <FileText className="text-primary-600 dark:text-primary-400" size={24} />
                    <div>
                      <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                        {selectedPrompt.title}
                      </h2>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        Category: {selectedPrompt.category} • Created: {new Date(selectedPrompt.created_at).toLocaleDateString()}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => toggleFavorite(selectedPrompt)}
                      className={`p-2 rounded-lg ${selectedPrompt.is_favorite ? 'text-yellow-500' : 'text-gray-400 hover:text-yellow-500'}`}
                    >
                      <Star size={20} className={selectedPrompt.is_favorite ? 'fill-current' : ''} />
                    </button>
                    <button
                      onClick={() => copyToClipboard(selectedPrompt.content)}
                      className="p-2 rounded-lg text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                    >
                      <Copy size={20} />
                    </button>
                    <button
                      onClick={() => handleEdit(selectedPrompt)}
                      className="btn btn-secondary text-sm"
                    >
                      <Edit size={16} className="mr-1" />
                      Edit
                    </button>
                    <button
                      onClick={() => handleDelete(selectedPrompt.id)}
                      className="btn btn-secondary text-sm text-red-600 hover:text-red-700"
                    >
                      <Trash2 size={16} className="mr-1" />
                      Delete
                    </button>
                  </div>
                </div>

                {/* Prompt Content */}
                <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
                  <pre className="whitespace-pre-wrap text-sm text-gray-900 dark:text-gray-100 font-mono">
                    {selectedPrompt.content}
                  </pre>
                </div>

                {/* Tags */}
                {selectedPrompt.tags && selectedPrompt.tags.length > 0 && (
                  <div className="mt-4">
                    <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Tags</h4>
                    <div className="flex flex-wrap gap-2">
                      {selectedPrompt.tags.map((tag, index) => (
                        <span key={index} className="px-2 py-1 text-xs bg-primary-100 text-primary-800 dark:bg-primary-900 dark:text-primary-200 rounded">
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Prompt Tester */}
              <div className="card">
                <h3 className="text-lg font-semibold mb-4">Test Prompt</h3>

                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium mb-2">Test Input (Optional)</label>
                    <textarea
                      value={testInput}
                      onChange={(e) => setTestInput(e.target.value)}
                      className="input w-full"
                      rows={3}
                      placeholder="Enter test input to combine with the prompt..."
                    />
                  </div>

                  <button
                    onClick={() => {
                      const combined = testInput
                        ? `${selectedPrompt.content}\n\nInput: ${testInput}`
                        : selectedPrompt.content
                      setTestOutput(combined)
                      copyToClipboard(combined)
                    }}
                    className="btn btn-primary"
                  >
                    <Play size={16} className="mr-2" />
                    Generate Test Output
                  </button>

                  {testOutput && (
                    <div>
                      <label className="block text-sm font-medium mb-2">Generated Output</label>
                      <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
                        <pre className="whitespace-pre-wrap text-sm text-gray-900 dark:text-gray-100">
                          {testOutput}
                        </pre>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div className="card">
              <div className="text-center py-12">
                <FileText size={48} className="mx-auto mb-4 text-gray-400" />
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                  Select a prompt to view
                </h3>
                <p className="text-gray-600 dark:text-gray-400">
                  Choose a prompt from the list to view its content and test it
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function getCategoryColor(category) {
  const colors = {
    general: 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200',
    coding: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
    writing: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
    analysis: 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200',
    creative: 'bg-pink-100 text-pink-800 dark:bg-pink-900 dark:text-pink-200'
  }
  return colors[category] || colors.general
}

export default PromptEngineeringStudio