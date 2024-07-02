import pandas as pd
import requests
import pytest

@pytest.fixture
def get_test_data():
    file_path = 'data/raw/test.csv'
    data = pd.read_csv(file_path, nrows=3)
    features = data.drop(columns=['ID', 'default payment next month'])
    expected_labels = data['default payment next month'].values
    return features, expected_labels

def test_api_predictions(get_test_data):
    features, expected_labels = get_test_data

    data_records = features.to_dict(orient='records')

    api_url = 'http://localhost:5000/predict'

    response = requests.post(api_url, json={'records': data_records})

    assert response.status_code == 200, f'Error: {response.status_code}, {response.text}'

    predictions = response.json()
    predicted_labels = predictions['predictions']
    
    correct_predictions = (predicted_labels == expected_labels).sum()
    accuracy = correct_predictions / len(expected_labels)

    assert accuracy > 0.5, f'Accuracy is too low: {accuracy * 100:.2f}%'

if __name__ == '__main__':
    pytest.main()
