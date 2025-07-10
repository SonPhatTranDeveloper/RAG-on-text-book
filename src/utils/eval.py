import json


def load_eval_dataset(dataset_path: str, grade: str, subject: str) -> list[dict]:
    """
    Load evaluation dataset from a file.
    """
    dataset_path = f"src/eval/data/{grade}/{subject}/dataset.json"
    with open(dataset_path, "r") as f:
        return json.load(f)
    

def load_eval_dataset_by_grade(grade: str) -> list[dict]:
    """
    Load evaluation dataset by grade, combining all subjects into one dataset.
    """
    import os
    
    combined_dataset = []
    
    # Iterate through all subjects and load their datasets
    for subject in os.listdir(f"src/eval/data/{grade}"):
        if os.path.isdir(f"src/eval/data/{grade}/{subject}"):
            subject_data = load_eval_dataset(dataset_path="", grade=grade, subject=subject)
            combined_dataset.extend(subject_data)
    
    return combined_dataset