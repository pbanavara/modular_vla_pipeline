import os
import json
import csv
from pathlib import Path

# Define mappings of keywords to categories
label_mappings = {
    "dirty": "dirty",
    "clean": "clean",
    "food_residue": "food residue",
    "food_stains": "food stains",
    "dutch_oven": "dutch oven",
    "plate": "plate",
    "pan": "pan",
    "cup": "cup",
    "mug": "mug",
    "spoon": "spoon",
    "fork": "fork",
    "coffee": "coffee",
    "cup": "cup",
    "casserole": "casserole",
    "wok": "wok",
    "saucepan": "saucepan",
    "saute pan": "saute pan",
    "sauce pan": "sauce pan",
    "burnt": "burnt",
    "pan": "pan",
    "cups": "cups",
    "stains": "stains",
    "stain": "stain",
    "residue": "residue",
    "food": "food",
    "oven": "oven",
    "dutch": "dutch",
}


def extract_labels(filename):
    """
    Extracts labels from a given filename using predefined mappings.
    """
    labels = set()
    words = filename.lower().split("_")

    for word in words:
        if word in label_mappings:
            labels.add(label_mappings[word])

    return list(labels)


def process_files(file_list):
    """
    Processes a list of filenames and assigns labels.
    """
    labeled_data = []

    for file in file_list:
        file_name = os.path.splitext(file)[0]  # Remove extension
        labels = extract_labels(file_name)
        labeled_data.append({"filename": file, "labels": labels})

    return labeled_data


def get_all_image_filenames(directory, extensions=("jpg", "jpeg", "png")):
    """
    Recursively lists all image files in the given directory.

    Args:
        directory (str): The path to the directory.
        extensions (tuple): Allowed image file extensions.

    Returns:
        List[str]: A list of image filenames.
    """
    directory = Path(directory)
    return [
        str(file.name)
        for file in directory.rglob("*")
        if file.suffix.lower().lstrip(".") in extensions
    ]

def main(image_dir):
    # Example usage
    file_list = get_all_image_filenames(image_dir)
    print(file_list)  # Prints the list of image filenames

    # Process files
    labeled_data = process_files(file_list)

    # Save to JSON
    with open("labeled_data.json", "w") as json_file:
        json.dump(labeled_data, json_file, indent=4)

    # Save to CSV
    csv_filename = "labeled_data.csv"
    with open(csv_filename, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["filename", "labels"])
        writer.writeheader()
        for row in labeled_data:
            writer.writerow(
                {"filename": row["filename"], "labels": ", ".join(row["labels"])}
            )

    print("Labeling completed! Data saved to labeled_data.json and labeled_data.csv")

if __name__ == "__main__":
    image_dir = input("Enter the path to your images directory: ")
    if not os.path.exists(image_dir):
        print("Directory does not exist. Exiting.")
        exit()
    main(image_dir=image_dir)