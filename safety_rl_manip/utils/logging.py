import json, os

def load_experiment_data(filename='experiment_results.json'):
    if not os.path.exists(filename):
        data={}
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)
    
    with open(filename, 'r') as file:
        return json.load(file)

def save_experiment_data(data=None, filename='experiment_status.json'):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def log_experiment_status(experiment_id=0, success=None, metrics=None, filename='experiment_status.json'):
    data = load_experiment_data(filename)
    data[experiment_id] = {"Success": success}
    if metrics:
        data[experiment_id]["metrics"] = metrics
    save_experiment_data(data, filename)