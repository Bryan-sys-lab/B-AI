import pytest
from fastapi.testclient import TestClient
from orchestrator.main import app

def test_project_task_workflow():
    """Test the complete workflow of creating a project, adding tasks, and retrieving them"""
    client = TestClient(app)

    # 1. Create a project
    project_response = client.post("/api/projects", json={
        "name": "Integration Test Project",
        "description": "Testing the full workflow"
    })
    assert project_response.status_code == 200
    project_data = project_response.json()
    project_id = project_data["id"]
    assert project_data["task_count"] == 0

    # 2. Verify project appears in list
    list_response = client.get("/api/projects")
    assert list_response.status_code == 200
    projects = list_response.json()
    assert any(p["id"] == project_id for p in projects)

    # 3. Create a task associated with the project
    task_response = client.post("/api/tasks", json={
        "description": "Integration test task",
        "project_id": project_id
    })
    assert task_response.status_code == 200
    task_data = task_response.json()
    task_id = task_data["task_id"]

    # 4. Verify task is associated with project
    get_task_response = client.get(f"/api/tasks/{task_id}")
    assert get_task_response.status_code == 200
    task_details = get_task_response.json()
    assert task_details["project_id"] == project_id
    assert task_details["description"] == "Integration test task"

    # 5. Verify project task count is updated
    updated_project_response = client.get(f"/api/projects/{project_id}")
    assert updated_project_response.status_code == 200
    updated_project = updated_project_response.json()
    assert updated_project["task_count"] >= 1

    # 6. List tasks filtered by project
    filtered_tasks_response = client.get(f"/api/tasks?project_id={project_id}")
    assert filtered_tasks_response.status_code == 200
    filtered_tasks = filtered_tasks_response.json()
    assert len(filtered_tasks) >= 1
    assert all(t["project_id"] == project_id for t in filtered_tasks)

    # 7. Create another task without project association
    standalone_task_response = client.post("/api/tasks", json={
        "description": "Standalone task"
    })
    assert standalone_task_response.status_code == 200

    # 8. Verify standalone task doesn't appear in project filter
    project_tasks_response = client.get(f"/api/tasks?project_id={project_id}")
    project_tasks = project_tasks_response.json()
    standalone_task_found = any(t["description"] == "Standalone task" for t in project_tasks)
    assert not standalone_task_found

    # 9. Submit feedback on the task
    feedback_response = client.post("/api/feedback", json={
        "task_id": task_id,
        "rating": 4,
        "comments": "Good work on the integration test",
        "improvement_suggestions": {"documentation": "Add more examples"}
    })
    assert feedback_response.status_code == 200

def test_error_handling():
    """Test error handling for invalid operations"""
    client = TestClient(app)

    # Try to get non-existent project
    response = client.get("/api/projects/invalid-id")
    assert response.status_code == 404

    # Try to get non-existent task
    response = client.get("/api/tasks/invalid-id")
    assert response.status_code == 404

    # Try to create task with invalid project
    response = client.post("/api/tasks", json={
        "description": "Task with invalid project",
        "project_id": "invalid-project-id"
    })
    assert response.status_code == 404

def test_task_listing_and_filtering():
    """Test various task listing and filtering scenarios"""
    client = TestClient(app)

    # Create multiple projects and tasks
    project1_response = client.post("/api/projects", json={"name": "Project 1"})
    project1_id = project1_response.json()["id"]

    project2_response = client.post("/api/projects", json={"name": "Project 2"})
    project2_id = project2_response.json()["id"]

    # Create tasks with different statuses and projects
    client.post("/api/tasks", json={"description": "Task 1", "project_id": project1_id})
    client.post("/api/tasks", json={"description": "Task 2", "project_id": project1_id})
    client.post("/api/tasks", json={"description": "Task 3", "project_id": project2_id})
    client.post("/api/tasks", json={"description": "Standalone Task"})

    # Test unfiltered list
    all_tasks_response = client.get("/api/tasks")
    all_tasks = all_tasks_response.json()
    assert len(all_tasks) >= 4

    # Test project filtering
    project1_tasks_response = client.get(f"/api/tasks?project_id={project1_id}")
    project1_tasks = project1_tasks_response.json()
    assert len(project1_tasks) >= 2
    assert all(t["project_id"] == project1_id for t in project1_tasks)

    project2_tasks_response = client.get(f"/api/tasks?project_id={project2_id}")
    project2_tasks = project2_tasks_response.json()
    assert len(project2_tasks) >= 1
    assert all(t["project_id"] == project2_id for t in project2_tasks)

    # Test status filtering (would need actual status updates to test fully)
    # This tests the API structure
    status_response = client.get("/api/tasks?status=pending")
    assert status_response.status_code == 200