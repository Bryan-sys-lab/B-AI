import { useState, useEffect } from 'react'
import { Brain, Code, AlertTriangle, CheckCircle, TrendingUp, RefreshCw, BarChart3 } from 'lucide-react'
import { intelligenceAPI } from '../services/api'
import { useNotifications } from '../providers/NotificationProvider'

function CodeIntelligenceTools() {
  const [analyses, setAnalyses] = useState([])
  const [loading, setLoading] = useState(true)
  const [selectedAnalysis, setSelectedAnalysis] = useState(null)
  const [showAnalysisForm, setShowAnalysisForm] = useState(false)
  const [analysisTarget, setAnalysisTarget] = useState('')
  const [analysisType, setAnalysisType] = useState('complexity')
  const { showSuccess, showError } = useNotifications()

  useEffect(() => {
    loadAnalyses()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const loadAnalyses = async () => {
    try {
      const response = await intelligenceAPI.list()
      setAnalyses(response.data)
    } catch (error) {
      showError('Failed to load code analyses')
    } finally {
      setLoading(false)
    }
  }

  const runAnalysis = async () => {
    if (!analysisTarget.trim()) return

    try {
      await intelligenceAPI.analyze({
        target: analysisTarget,
        type: analysisType
      })
      showSuccess('Code analysis started')
      setShowAnalysisForm(false)
      setAnalysisTarget('')
      loadAnalyses()
    } catch (error) {
      showError('Failed to start code analysis')
    }
  }

  const getAnalysisStatusColor = (status) => {
    switch (status) {
      case 'completed': return 'text-green-600 dark:text-green-400'
      case 'running': return 'text-blue-600 dark:text-blue-400'
      case 'failed': return 'text-red-600 dark:text-red-400'
      default: return 'text-gray-600 dark:text-gray-400'
    }
  }

  const getAnalysisStatusIcon = (status) => {
    switch (status) {
      case 'completed': return <CheckCircle size={16} />
      case 'running': return <RefreshCw size={16} className="animate-spin" />
      case 'failed': return <AlertTriangle size={16} />
      default: return <RefreshCw size={16} />
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
          Code Intelligence Tools
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Advanced code analysis, complexity metrics, and intelligent insights
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Analysis Tools */}
        <div className="lg:col-span-1">
          <div className="card">
            <h2 className="text-xl font-semibold mb-4">Analysis Tools</h2>

            <div className="space-y-4">
              <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
                <div className="flex items-center mb-2">
                  <Code className="text-blue-600 dark:text-blue-400 mr-2" size={20} />
                  <h3 className="font-medium">Complexity Analysis</h3>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                  Analyze code complexity, maintainability, and quality metrics
                </p>
                <button
                  onClick={() => {
                    setAnalysisType('complexity')
                    setShowAnalysisForm(true)
                  }}
                  className="btn btn-primary w-full text-sm"
                >
                  Run Analysis
                </button>
              </div>

              <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
                <div className="flex items-center mb-2">
                  <BarChart3 className="text-green-600 dark:text-green-400 mr-2" size={20} />
                  <h3 className="font-medium">Performance Analysis</h3>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                  Identify performance bottlenecks and optimization opportunities
                </p>
                <button
                  onClick={() => {
                    setAnalysisType('performance')
                    setShowAnalysisForm(true)
                  }}
                  className="btn btn-primary w-full text-sm"
                >
                  Run Analysis
                </button>
              </div>

              <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
                <div className="flex items-center mb-2">
                  <Brain className="text-purple-600 dark:text-purple-400 mr-2" size={20} />
                  <h3 className="font-medium">Security Analysis</h3>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                  Detect security vulnerabilities and best practice violations
                </p>
                <button
                  onClick={() => {
                    setAnalysisType('security')
                    setShowAnalysisForm(true)
                  }}
                  className="btn btn-primary w-full text-sm"
                >
                  Run Analysis
                </button>
              </div>

              <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
                <div className="flex items-center mb-2">
                  <TrendingUp className="text-orange-600 dark:text-orange-400 mr-2" size={20} />
                  <h3 className="font-medium">Quality Metrics</h3>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                  Comprehensive code quality and maintainability assessment
                </p>
                <button
                  onClick={() => {
                    setAnalysisType('quality')
                    setShowAnalysisForm(true)
                  }}
                  className="btn btn-primary w-full text-sm"
                >
                  Run Analysis
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Analysis Form */}
        <div className="lg:col-span-2">
          {showAnalysisForm ? (
            <div className="card">
              <h2 className="text-xl font-semibold mb-4">
                Run {analysisType.charAt(0).toUpperCase() + analysisType.slice(1)} Analysis
              </h2>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Target Path or URL</label>
                  <input
                    type="text"
                    value={analysisTarget}
                    onChange={(e) => setAnalysisTarget(e.target.value)}
                    placeholder="e.g., /src/components, https://github.com/user/repo"
                    className="input w-full"
                  />
                </div>

                <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                  <h4 className="font-medium text-blue-900 dark:text-blue-100 mb-2">
                    What this analysis will do:
                  </h4>
                  <ul className="text-sm text-blue-800 dark:text-blue-200 space-y-1">
                    {analysisType === 'complexity' && (
                      <>
                        <li>• Calculate cyclomatic complexity</li>
                        <li>• Analyze code maintainability index</li>
                        <li>• Identify overly complex functions</li>
                        <li>• Suggest refactoring opportunities</li>
                      </>
                    )}
                    {analysisType === 'performance' && (
                      <>
                        <li>• Analyze algorithmic complexity</li>
                        <li>• Identify performance bottlenecks</li>
                        <li>• Suggest optimization strategies</li>
                        <li>• Memory usage analysis</li>
                      </>
                    )}
                    {analysisType === 'security' && (
                      <>
                        <li>• Scan for common vulnerabilities</li>
                        <li>• Check for insecure patterns</li>
                        <li>• Validate input sanitization</li>
                        <li>• Review authentication/authorization</li>
                      </>
                    )}
                    {analysisType === 'quality' && (
                      <>
                        <li>• Code coverage analysis</li>
                        <li>• Documentation completeness</li>
                        <li>• Test quality assessment</li>
                        <li>• Code style consistency</li>
                      </>
                    )}
                  </ul>
                </div>

                <div className="flex space-x-2">
                  <button onClick={runAnalysis} className="btn btn-primary">
                    Start Analysis
                  </button>
                  <button
                    onClick={() => setShowAnalysisForm(false)}
                    className="btn btn-secondary"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            </div>
          ) : selectedAnalysis ? (
            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold">
                  Analysis Results: {selectedAnalysis.type}
                </h2>
                <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                  selectedAnalysis.status === 'completed' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' :
                  selectedAnalysis.status === 'running' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200' :
                  'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                }`}>
                  {selectedAnalysis.status}
                </span>
              </div>

              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                      Target
                    </label>
                    <p className="text-sm">{selectedAnalysis.target}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                      Started
                    </label>
                    <p className="text-sm">{new Date(selectedAnalysis.created_at).toLocaleString()}</p>
                  </div>
                </div>

                {selectedAnalysis.results && (
                  <div>
                    <h3 className="text-lg font-medium mb-3">Analysis Results</h3>

                    {selectedAnalysis.type === 'complexity' && (
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                          <h4 className="font-medium mb-2">Complexity Metrics</h4>
                          <div className="space-y-2 text-sm">
                            <div className="flex justify-between">
                              <span>Average Complexity:</span>
                              <span className="font-medium">{selectedAnalysis.results.avgComplexity || 'N/A'}</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Max Complexity:</span>
                              <span className="font-medium">{selectedAnalysis.results.maxComplexity || 'N/A'}</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Maintainability Index:</span>
                              <span className="font-medium">{selectedAnalysis.results.maintainabilityIndex || 'N/A'}</span>
                            </div>
                          </div>
                        </div>

                        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                          <h4 className="font-medium mb-2">Issues Found</h4>
                          <div className="space-y-2">
                            {selectedAnalysis.results.issues?.map((issue, index) => (
                              <div key={index} className="flex items-center justify-between text-sm">
                                <span>{issue.description}</span>
                                <span className={`px-2 py-1 text-xs rounded ${
                                  issue.severity === 'high' ? 'bg-red-100 text-red-800' :
                                  issue.severity === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                                  'bg-green-100 text-green-800'
                                }`}>
                                  {issue.severity}
                                </span>
                              </div>
                            )) || <p className="text-sm text-gray-500">No issues found</p>}
                          </div>
                        </div>
                      </div>
                    )}

                    {selectedAnalysis.type === 'performance' && (
                      <div className="space-y-4">
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                          <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg text-center">
                            <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                              {selectedAnalysis.results.optimizations || 0}
                            </div>
                            <div className="text-sm text-green-800 dark:text-green-200">Optimizations Found</div>
                          </div>
                          <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg text-center">
                            <div className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">
                              {selectedAnalysis.results.bottlenecks || 0}
                            </div>
                            <div className="text-sm text-yellow-800 dark:text-yellow-200">Bottlenecks Identified</div>
                          </div>
                          <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg text-center">
                            <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                              {selectedAnalysis.results.score || 0}%
                            </div>
                            <div className="text-sm text-blue-800 dark:text-blue-200">Performance Score</div>
                          </div>
                        </div>
                      </div>
                    )}

                    {selectedAnalysis.type === 'security' && (
                      <div className="space-y-4">
                        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                          <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg text-center">
                            <div className="text-2xl font-bold text-red-600 dark:text-red-400">
                              {selectedAnalysis.results.critical || 0}
                            </div>
                            <div className="text-sm text-red-800 dark:text-red-200">Critical Issues</div>
                          </div>
                          <div className="p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg text-center">
                            <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
                              {selectedAnalysis.results.high || 0}
                            </div>
                            <div className="text-sm text-orange-800 dark:text-orange-200">High Risk</div>
                          </div>
                          <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg text-center">
                            <div className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">
                              {selectedAnalysis.results.medium || 0}
                            </div>
                            <div className="text-sm text-yellow-800 dark:text-yellow-200">Medium Risk</div>
                          </div>
                          <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg text-center">
                            <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                              {selectedAnalysis.results.score || 0}%
                            </div>
                            <div className="text-sm text-green-800 dark:text-green-200">Security Score</div>
                          </div>
                        </div>
                      </div>
                    )}

                    {selectedAnalysis.type === 'quality' && (
                      <div className="space-y-4">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                            <h4 className="font-medium mb-2">Code Coverage</h4>
                            <div className="text-2xl font-bold text-blue-600 dark:text-blue-400 mb-1">
                              {selectedAnalysis.results.coverage || 0}%
                            </div>
                            <div className="w-full bg-blue-200 dark:bg-blue-800 rounded-full h-2">
                              <div
                                className="bg-blue-600 h-2 rounded-full"
                                style={{ width: `${selectedAnalysis.results.coverage || 0}%` }}
                              />
                            </div>
                          </div>

                          <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                            <h4 className="font-medium mb-2">Documentation</h4>
                            <div className="text-2xl font-bold text-purple-600 dark:text-purple-400 mb-1">
                              {selectedAnalysis.results.documentation || 0}%
                            </div>
                            <div className="w-full bg-purple-200 dark:bg-purple-800 rounded-full h-2">
                              <div
                                className="bg-purple-600 h-2 rounded-full"
                                style={{ width: `${selectedAnalysis.results.documentation || 0}%` }}
                              />
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="card">
              <div className="text-center py-12">
                <Brain size={48} className="mx-auto mb-4 text-gray-400" />
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                  Select an analysis tool
                </h3>
                <p className="text-gray-600 dark:text-gray-400">
                  Choose an analysis type from the left panel to get started
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Recent Analyses */}
      <div className="mt-8">
        <div className="card">
          <h2 className="text-xl font-semibold mb-4">Recent Analyses</h2>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left py-2 px-4 font-medium">Type</th>
                  <th className="text-left py-2 px-4 font-medium">Target</th>
                  <th className="text-left py-2 px-4 font-medium">Status</th>
                  <th className="text-left py-2 px-4 font-medium">Started</th>
                  <th className="text-left py-2 px-4 font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {analyses.slice(0, 10).map((analysis) => (
                  <tr
                    key={analysis.id}
                    className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800 cursor-pointer"
                    onClick={() => setSelectedAnalysis(analysis)}
                  >
                    <td className="py-2 px-4 capitalize">{analysis.type}</td>
                    <td className="py-2 px-4 truncate max-w-xs">{analysis.target}</td>
                    <td className="py-2 px-4">
                      <div className={`flex items-center ${getAnalysisStatusColor(analysis.status)}`}>
                        {getAnalysisStatusIcon(analysis.status)}
                        <span className="ml-1 capitalize">{analysis.status}</span>
                      </div>
                    </td>
                    <td className="py-2 px-4">{new Date(analysis.created_at).toLocaleDateString()}</td>
                    <td className="py-2 px-4">
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          setSelectedAnalysis(analysis)
                        }}
                        className="text-primary-600 hover:text-primary-700 dark:text-primary-400 dark:hover:text-primary-300"
                      >
                        View
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {analyses.length === 0 && (
            <div className="text-center py-8 text-gray-500 dark:text-gray-400">
              <Brain size={32} className="mx-auto mb-2 opacity-50" />
              <p>No analyses have been run yet</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default CodeIntelligenceTools