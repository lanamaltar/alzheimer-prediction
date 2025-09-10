import pytest
from flask import json
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_register_and_login(client):
    # Register
    response = client.post('/register', json={
        'username': 'apitestuser',
        'password': 'apitestpass123'
    })
    assert response.status_code in (201, 409)  # 409 if user already exists
    # Login
    response = client.post('/login', json={
        'username': 'apitestuser',
        'password': 'apitestpass123'
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data['success']
    assert 'token' in data['data']
    return data['data']['token']

def test_numeric_prediction(client):
    token = test_register_and_login(client)
    response = client.post('/predict/numeric',
        json={'input': [23, 1, 0, 28, 2, 1, 0, 1, 0, 1]},
        headers={'Authorization': f'Bearer {token}'}
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data['success']
    assert 'prediction' in data['data']

def test_history(client):
    token = test_register_and_login(client)
    response = client.get('/history', headers={'Authorization': f'Bearer {token}'})
    assert response.status_code == 200
    data = response.get_json()
    assert data['success']
    assert 'history' in data['data']
