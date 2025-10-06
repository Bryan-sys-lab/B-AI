import { useState, useEffect } from 'react'
import { Plus, Edit, Trash2, FolderOpen, Users, Calendar, RefreshCw } from 'lucide-react'
import { projectAPI } from '../services/api'
import { useNotifications } from '../providers/NotificationProvider'

function ProjectManagement() {
  const [projects, setProjects] = useState([])
  const [loading, setLoading] = useState(true)
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [editingProject, setEditingProject] = useState(null)
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    repository_url: '',
    status: 'active'
  })
  const { showSuccess, showError } = useNotifications()

  useEffect(() => {
    loadProjects()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const loadProjects = async () => {
    try {
      const response = await projectAPI.list()
      setProjects(response.data)
    } catch (error) {
      showError('Failed to load projects')
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    try {
      if (editingProject) {
        // Update existing project
        await projectAPI.update(editingProject.id, formData)
        showSuccess('Project updated successfully')
      } else {
        // Create new project
        await projectAPI.create(formData)
        showSuccess('Project created successfully')
      }
      resetForm()
      loadProjects()
    } catch (error) {
      showError(`Failed to ${editingProject ? 'update' : 'create'} project`)
    }
  }

  const handleEdit = (project) => {
    setEditingProject(project)
    setFormData({
      name: project.name,
      description: project.description,
      repository_url: project.repository_url || '',
      status: project.status
    })
    setShowCreateForm(true)
  }

  const handleDelete = async (projectId) => {
    if (!confirm('Are you sure you want to delete this project?')) return

    try {
      await projectAPI.delete(projectId)
      showSuccess('Project deleted successfully')
      loadProjects()
    } catch (error) {
      showError('Failed to delete project')
    }
  }

  const resetForm = () => {
    setFormData({
      name: '',
      description: '',
      repository_url: '',
      status: 'active'
    })
    setEditingProject(null)
    setShowCreateForm(false)
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'active': return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
      case 'completed': return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
      case 'on-hold': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
      case 'cancelled': return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
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
    <div className="max-w-7xl mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          Project Management
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Create and manage development projects with team collaboration
        </p>
      </div>

      <div className="mb-6">
        <button
          onClick={() => setShowCreateForm(true)}
          className="btn btn-primary"
        >
          <Plus size={16} className="mr-2" />
          New Project
        </button>
      </div>

      {showCreateForm && (
        <div className="card mb-6">
          <h2 className="text-xl font-semibold mb-4">
            {editingProject ? 'Edit Project' : 'Create New Project'}
          </h2>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1">Project Name</label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({...formData, name: e.target.value})}
                  className="input w-full"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Status</label>
                <select
                  value={formData.status}
                  onChange={(e) => setFormData({...formData, status: e.target.value})}
                  className="input w-full"
                >
                  <option value="active">Active</option>
                  <option value="completed">Completed</option>
                  <option value="on-hold">On Hold</option>
                  <option value="cancelled">Cancelled</option>
                </select>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Description</label>
              <textarea
                value={formData.description}
                onChange={(e) => setFormData({...formData, description: e.target.value})}
                className="input w-full"
                rows={3}
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Repository URL (Optional)</label>
              <input
                type="url"
                value={formData.repository_url}
                onChange={(e) => setFormData({...formData, repository_url: e.target.value})}
                className="input w-full"
                placeholder="https://github.com/user/repo"
              />
            </div>

            <div className="flex space-x-2">
              <button type="submit" className="btn btn-primary">
                {editingProject ? 'Update Project' : 'Create Project'}
              </button>
              <button type="button" onClick={resetForm} className="btn btn-secondary">
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {projects.map((project) => (
          <div key={project.id} className="card">
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center">
                <FolderOpen className="text-primary-600 dark:text-primary-400 mr-2" size={24} />
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  {project.name}
                </h3>
              </div>
              <span className={`px-2 py-1 text-xs font-medium rounded-full capitalize ${getStatusColor(project.status)}`}>
                {project.status}
              </span>
            </div>

            <p className="text-gray-600 dark:text-gray-400 mb-4 line-clamp-3">
              {project.description}
            </p>

            <div className="space-y-2 mb-4">
              {project.repository_url && (
                <div className="flex items-center text-sm text-gray-600 dark:text-gray-400">
                  <Users size={14} className="mr-2" />
                  <a
                    href={project.repository_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:text-primary-600 dark:hover:text-primary-400"
                  >
                    Repository
                  </a>
                </div>
              )}

              <div className="flex items-center text-sm text-gray-600 dark:text-gray-400">
                <Calendar size={14} className="mr-2" />
                Created {new Date(project.created_at).toLocaleDateString()}
              </div>
            </div>

            <div className="flex space-x-2">
              <button
                onClick={() => handleEdit(project)}
                className="btn btn-secondary flex-1 text-sm"
              >
                <Edit size={14} className="mr-1" />
                Edit
              </button>
              <button
                onClick={() => handleDelete(project.id)}
                className="btn btn-secondary flex-1 text-sm text-red-600 hover:text-red-700"
              >
                <Trash2 size={14} className="mr-1" />
                Delete
              </button>
            </div>
          </div>
        ))}
      </div>

      {projects.length === 0 && (
        <div className="text-center py-12">
          <FolderOpen size={48} className="mx-auto mb-4 text-gray-400" />
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
            No projects yet
          </h3>
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            Create your first project to get started
          </p>
          <button
            onClick={() => setShowCreateForm(true)}
            className="btn btn-primary"
          >
            <Plus size={16} className="mr-2" />
            Create Project
          </button>
        </div>
      )}
    </div>
  )
}

export default ProjectManagement