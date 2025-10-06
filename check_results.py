import httpx
import asyncio
import json

async def check_task_results():
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get('http://localhost:8000/api/tasks')
            if response.status_code == 200:
                data = response.json()
                tasks = data.get('tasks', [])

                print(f'Found {len(tasks)} tasks')

                for i, task in enumerate(tasks[-3:]):
                    print(f'\n--- Task {i+1}: {task["id"]} ---')
                    print(f'Status: {task["status"]}')
                    print(f'Description: {task["description"][:80]}...')

                    task_response = await client.get(f'http://localhost:8000/api/tasks/{task["id"]}')
                    if task_response.status_code == 200:
                        task_details = task_response.json()
                        output = task_details.get('output', {})
                        if isinstance(output, str):
                            try:
                                output = json.loads(output)
                            except:
                                pass

                        if output:
                            if isinstance(output, dict) and 'response' in output:
                                response_text = output['response']
                                print(f'Output length: {len(response_text)} chars')
                                print(f'Output preview: {response_text[:200]}...')

                                if '```' in response_text:
                                    print('✓ Contains code blocks')
                                else:
                                    print('✗ No code blocks found')
                            else:
                                print(f'Output type: {type(output)}')
                        else:
                            print('No output available')
                    else:
                        print(f'Failed to get task details: {task_response.status_code}')
            else:
                print(f'Failed to get tasks: {response.status_code}')
    except Exception as e:
        print(f'Error: {e}')

if __name__ == "__main__":
    asyncio.run(check_task_results())