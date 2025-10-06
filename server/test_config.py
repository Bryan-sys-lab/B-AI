import json

try:
    with open('new_nim_config.json', 'r') as f:
        data = json.load(f)
    print('Config loaded successfully')
    print('Keys:', list(data.keys()))
    print('Role mapping keys:', list(data['role_mapping_without_openrouter'].keys()))
except Exception as e:
    print(f'Error: {e}')