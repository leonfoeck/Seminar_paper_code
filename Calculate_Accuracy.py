import json


def compute_top5_accuracy(filename):
    total_count = 0
    correct_count = 0

    with open(filename, 'r') as file:
        for line in file:
            data = json.loads(line)
            predictions = data['predictions'][0]
            references = data['references'][0]

            # Increment the total count for every line in the file
            total_count += 1

            # Check if any of the references appear in the top 5 predictions
            if references == predictions:
                correct_count += 1

    return correct_count / total_count


filename = "bigDataSet_10Epochs_beam_Top_1.json"
top5_accuracy = compute_top5_accuracy(filename)
print(f"Top-5 Accuracy: {top5_accuracy * 100:.2f}%")
