#!/usr/bin/env python3
"""
Simple test script to verify conversation history updates work correctly.
"""
import asyncio
import sys
import os

# Add server to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'server'))

from orchestrator.database import init_db, async_session, Task
from orchestrator.workflow import orchestrate_task_flow
from sqlalchemy import select

async def test_conversation_history():
    """Test that conversation history is properly updated after task completion."""
    print("Initializing database...")
    await init_db()

    async with async_session() as session:
        # Create a test task
        test_task = Task(
            user_id="test_user_123",
            description="Hello, can you help me with a simple Python script?",
            context={"conversation_history": []}
        )
        session.add(test_task)
        await session.commit()
        task_id = test_task.id
        print(f"Created test task with ID: {task_id}")

    print("Running orchestration...")
    try:
        # This will fail because agents aren't running, but we can check if conversation history gets updated
        result = await orchestrate_task_flow(task_id)
    except Exception as e:
        print(f"Expected error (agents not running): {e}")

    # Check if conversation history was updated
    async with async_session() as session:
        result = await session.execute(select(Task).where(Task.id == task_id))
        task = result.scalar_one()

        print(f"Task context after orchestration: {task.context}")

        conversation_history = task.context.get("conversation_history", [])
        print(f"Conversation history length: {len(conversation_history)}")

        if conversation_history:
            print("SUCCESS: Conversation history was updated!")
            for i, msg in enumerate(conversation_history):
                print(f"  Message {i}: {msg['type']} - {msg['content'][:100]}...")
        else:
            print("FAILURE: Conversation history was not updated")

if __name__ == "__main__":
    asyncio.run(test_conversation_history())