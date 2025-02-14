import json

modes = ["leaves", "upper", "leaves_with_statuses", "upper_with_statuses"]

all_mappings = {}

for mode in modes:
    with open(f"{mode}_mapping.json", 'r') as f:
        mapping = json.load(f)
        labels = mapping.keys()
        all_mappings[mode] = {"mapping": mapping, "labels": labels}

