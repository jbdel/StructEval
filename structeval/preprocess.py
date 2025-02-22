import os
import glob
import argparse
import pandas as pd

##########################################################################
# Helpers
##########################################################################

def extract_leaves(label_list):
    """Extracts the leaf part of each label."""
    return [label.split(' (')[0] for label in label_list]

def map_to_upper(row, mapping):
    """Maps each leaf to its upper-level label using the provided mapping."""
    upper = []
    for leaf in row:
        if leaf in mapping:
            upper.append(mapping[leaf])
        else:
            print(f"Mapping not found for: {leaf}")
            upper.append(None)
    return upper

def map_to_upper_with_statuses(row, mapping):
    """Maps each label (with status) to its upper-level label with statuses.
       Returns None if conflicting statuses are found."""
    upper_status_map = {}
    for leaf_status in row:
        if leaf_status.lower() == "no finding":
            return ["No Finding"]

        if " (" in leaf_status:
            leaf, status = leaf_status.rsplit(' (', 1)
            status = status.strip(')')
        else:
            continue

        # Map leaf to upper-level label
        upper = mapping.get(leaf, leaf)

        # Check for conflicting statuses
        if upper in upper_status_map and upper_status_map[upper] != status:
            return None
        upper_status_map[upper] = status

    # Combine upper label and its status
    return [f"{upper} ({status})" for upper, status in upper_status_map.items()]

##########################################################################
# Process a single parquet file
##########################################################################

def process_file(filename, disease_mapping):
    print(f"Processing file: {filename}")
    df = pd.read_parquet(filename)

    # Extract leaves from the raw labels column
    df['leaves'] = df['labels'].apply(extract_leaves)
    df['leaves'] = df['leaves'].apply(lambda x: ['No Finding'] if any('no finding' in s.lower() for s in x) else x)

    # Rename and create columns for raw labels with statuses
    df.rename(columns={'labels': 'raw_labels'}, inplace=True)
    df['leaves_with_statuses'] = df['raw_labels']
    df['leaves_with_statuses'] = df['leaves_with_statuses'].apply(lambda x: ['No Finding'] if any('no finding' in s.lower() for s in x) else x)

    # Map leaves to upper-level labels and drop rows with missing mappings
    df['upper'] = df['leaves'].apply(lambda x: map_to_upper(x, disease_mapping))
    df = df[df['upper'].apply(lambda x: None not in x)]

    # Map with statuses and drop rows with conflicting statuses
    df['upper_with_statuses'] = df['leaves_with_statuses'].apply(lambda x: map_to_upper_with_statuses(x, disease_mapping))
    conflict_indices = df[df['upper_with_statuses'].isnull()].index
    df_processed = df.drop(index=conflict_indices)

    # Save the processed DataFrame as JSON
    output_filename = filename.replace('.parquet', '_processed.json')
    df_processed.reset_index(drop=True, inplace=True)
    df_processed.to_json(output_filename, orient="records", indent=4)
    print(f"DataFrame saved to {output_filename}")

##########################################################################
# Main
##########################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess all parquet files in a directory.")
    parser.add_argument("directory", type=str, help="Directory containing parquet files.")
    args = parser.parse_args()

    parquet_files = glob.glob(os.path.join(args.directory, '*.parquet'))
    if not parquet_files:
        print("No parquet files found in the specified directory.")
        exit(1)

    disease_mapping = {
        "No Finding": "No Finding",
        "Lung Lesion": "Lung Finding",
        "Fibrosis": "Lung Finding",
        "Emphysema": "Lung Finding",
        "Pulmonary congestion": "Lung Finding",
        "Bronchiectasis": "Lung Finding",
        "Lung Finding": "Lung Finding",
        "Hilar lymphadenopathy": "Lung Finding",
        "Diffuse air space opacity": "Air space opacity",
        "Air space opacity–multifocal": "Air space opacity",
        "Edema": "Diffuse air space opacity",
        "Consolidation": "Focal air space opacity",
        "Focal air space opacity": "Focal air space opacity",
        "Perihilar airspace opacity": "Focal air space opacity",
        "Pneumonia": "Consolidation",
        "Atelectasis": "Consolidation",
        "Aspiration": "Consolidation",
        "Segmental collapse": "Focal air space opacity",
        "Lung collapse": "Segmental collapse",
        "Solitary masslike opacity": "Masslike opacity",
        "Mass/Solitary lung mass": "Solitary masslike opacity",
        "Nodule/Solitary lung nodule": "Solitary masslike opacity",
        "Cavitating mass with content": "Solitary masslike opacity",
        "Multiple masslike opacities": "Masslike opacity",
        "Cavitating masses": "Multiple masslike opacities",
        "Pneumothorax": "Pleural finding",
        "Hydropneumothorax": "Pleural finding",
        "Pleural Other": "Pleural finding",
        "Simple pneumothorax": "Pneumothorax",
        "Loculated pneumothorax": "Pneumothorax",
        "Tension pneumothorax": "Pneumothorax",
        "Pleural Effusion": "Pleural Thickening",
        "Pleural scarring": "Pleural Thickening",
        "Simple pleural effusion": "Pleural Effusion",
        "Loculated pleural effusion": "Pleural Effusion",
        "Widened cardiac silhouette": "None",
        "Cardiomegaly": "Widened cardiac silhouette",
        "Pericardial effusion": "Widened cardiac silhouette",
        "Hernia": "Mediastinal finding",
        "Mediastinal mass": "Mediastinal finding",
        "Pneumomediastinum": "Mediastinal finding",
        "Tracheal deviation": "Mediastinal finding",
        "Inferior mediastinal mass": "Mediastinal mass",
        "Superior mediastinal mass": "Mediastinal mass",
        "Widened aortic contour": "Vascular finding",
        "Tortuous Aorta": "Widened aortic contour",
        "Fracture": "Musculoskeletal finding",
        "Shoulder dislocation": "Musculoskeletal finding",
        "Acute humerus fracture": "Fracture",
        "Acute rib fracture": "Fracture",
        "Acute clavicle fracture": "Fracture",
        "Acute scapula fracture": "Fracture",
        "Compression fracture": "Fracture",
        "Chest wall finding": "Musculoskeletal finding",
        "Subcutaneous Emphysema": "Chest wall finding",
        "Suboptimal central line": "Support Devices",
        "Suboptimal endotracheal tube": "Support Devices",
        "Suboptimal nasogastric tube": "Support Devices",
        "Suboptimal pulmonary arterial catheter": "Support Devices",
        "Pleural tube": "Support Devices",
        "PICC line": "Support Devices",
        "Port catheter": "Support Devices",
        "Pacemaker": "Support Devices",
        "Implantable defibrillator": "Support Devices",
        "LVAD": "Support Devices",
        "Intraaortic balloon pump": "Support Devices",
        "Subdiaphragmatic gas": "Upper abdominal finding",
        "Pneumoperitoneum": "Subdiaphragmatic gas",
        "Calcification of the Aorta": "Vascular finding",
        "Enlarged pulmonary artery": "Vascular finding",
    }

    for filename in parquet_files:
        process_file(filename, disease_mapping)
