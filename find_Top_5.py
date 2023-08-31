import json


def compute_top5_accuracy(filename):

    top5_Dict = {}

    with open(filename, 'r') as file:
        for line in file:
            data = json.loads(line)
            predictions = data['predictions']
            references = data['references'][0]
            if references in ["game", "mario", "easy", "sound"]:
                top5_Dict[references] = predictions

    return top5_Dict


filename = "bigDataSet_10Epochs_beam_Top_5.json"
top5_dict = compute_top5_accuracy(filename)
print(top5_dict)
