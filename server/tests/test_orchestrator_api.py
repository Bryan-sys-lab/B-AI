from fastapi.testclient import TestClient
from orchestrator.main import app

def test_create_project():
    """Test creating a new project"""
    client = TestClient(app)
    response = client.post("/api/projects", json={
        "name": "Test Project",
        "description": "A test project"
    })

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Test Project"
    assert data["description"] == "A test project"
    assert data["status"] == "active"
    assert "id" in data
    assert "created_at" in data

def test_list_projects():
    """Test listing projects"""
    client = TestClient(app)
    # First create a project
    create_response = client.post("/api/projects", json={
        "name": "List Test Project"
    })
    assert create_response.status_code == 200

    # Then list projects
    response = client.get("/api/projects")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1

    # Check the created project is in the list
    project_names = [p["name"] for p in data]
    assert "List Test Project" in project_names

def test_get_project():
    """Test getting a specific project"""
    client = TestClient(app)
    # Create a project first
    create_response = client.post("/api/projects", json={
        "name": "Get Test Project",
        "description": "Project for get test"
    })
    assert create_response.status_code == 200
    project_id = create_response.json()["id"]

    # Get the project
    response = client.get(f"/api/projects/{project_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == project_id
    assert data["name"] == "Get Test Project"
    assert data["description"] == "Project for get test"

def test_get_nonexistent_project():
    """Test getting a project that doesn't exist"""
    client = TestClient(app)
    response = client.get("/api/projects/nonexistent-id")
    assert response.status_code == 404
    assert "Project not found" in response.json()["detail"]

def test_create_task():
    """Test creating a new task"""
    client = TestClient(app)
    response = client.post("/api/tasks", json={
        "description": "Test task description"
    })

    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert isinstance(data["task_id"], str)

def test_create_task_with_project():
    """Test creating a task associated with a project"""
    client = TestClient(app)
    # Create a project first
    project_response = client.post("/api/projects", json={
        "name": "Task Project"
    })
    assert project_response.status_code == 200
    project_id = project_response.json()["id"]

    # Create a task with the project
    task_response = client.post("/api/tasks", json={
        "description": "Task with project",
        "project_id": project_id
    })
    assert task_response.status_code == 200

def test_list_tasks():
    """Test listing tasks"""
    client = TestClient(app)
    # Create a task first
    create_response = client.post("/api/tasks", json={
        "description": "List test task"
    })
    assert create_response.status_code == 200

    # List tasks
    response = client.get("/api/tasks")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1

    # Check task structure
    task = data[0]
    assert "id" in task
    assert "description" in task
    assert "status" in task
    assert "user_id" in task
    assert "created_at" in task

def test_list_tasks_by_project():
    """Test filtering tasks by project"""
    client = TestClient(app)
    # Create a project
    project_response = client.post("/api/projects", json={
        "name": "Filter Project"
    })
    project_id = project_response.json()["id"]

    # Create tasks - one with project, one without
    client.post("/api/tasks", json={
        "description": "Task with project",
        "project_id": project_id
    })
    client.post("/api/tasks", json={
        "description": "Task without project"
    })

    # List tasks for the project
    response = client.get(f"/api/tasks?project_id={project_id}")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["description"] == "Task with project"
    assert data[0]["project_id"] == project_id

def test_get_task():
    """Test getting a specific task"""
    client = TestClient(app)
    # Create a task
    create_response = client.post("/api/tasks", json={
        "description": "Get test task"
    })
    task_id = create_response.json()["task_id"]

    # Get the task
    response = client.get(f"/api/tasks/{task_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == task_id
    assert data["description"] == "Get test task"
    assert "subtasks" in data

def test_get_nonexistent_task():
    """Test getting a task that doesn't exist"""
    client = TestClient(app)
    response = client.get("/api/tasks/nonexistent-id")
    assert response.status_code == 404
    assert "Task not found" in response.json()["detail"]

def test_submit_feedback():
    """Test submitting feedback"""
    client = TestClient(app)
    # Create a task first
    task_response = client.post("/api/tasks", json={
        "description": "Feedback test task"
    })
    task_id = task_response.json()["task_id"]

    # Submit feedback
    feedback_response = client.post("/api/feedback", json={
        "task_id": task_id,
        "rating": 5,
        "comments": "Great work!",
        "improvement_suggestions": {"speed": "faster"}
    })
    assert feedback_response.status_code == 200
    data = feedback_response.json()
    assert "feedback_id" in data

def test_about_endpoint():
    """Test the about endpoint"""
    client = TestClient(app)
    response = client.get("/about")
    assert response.status_code == 200
    data = response.json()
    assert "level" in data
    assert "response" in data
    assert "system_prompt" in data