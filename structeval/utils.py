import warnings
import re

VALID_ORGANS = [
    "Lungs and Airways",
    "Pleura",
    "Cardiovascular",
    "Hila and Mediastinum",
    "Tubes, Catheters, and Support Devices",
    "Musculoskeletal and Chest Wall",
    "Abdominal",
    "Other",
]


def remove_bullets(findings):
    findings = [f[2:] if f[:2] == "- " else f for f in findings]
    return findings


def remove_numbering(impressions):
    """
    Remove numbering from a list of impressions.

    Parameters:
    impressions (list of str): List of impressions with numbering.

    Returns:
    list of str: List of impressions without numbering.
    """
    if not isinstance(impressions, list):
        raise TypeError("impressions should be of type list")

    new_impressions = []
    for imp in impressions:
        # Remove leading numbers, periods, and any following whitespace
        new_imp = re.sub(r'^\d+\.\s*', '', imp)
        new_impressions.append(new_imp)
    return new_impressions


def reconstruct_findings(findings_dict):
    if not isinstance(findings_dict, dict):
        raise TypeError("findings_dict should be of type dict")

    findings_str = ""
    for i, (organ_structure, organ_utterances) in enumerate(findings_dict.items()):
        findings_str += organ_structure + "\n"
        findings_str += "\n".join(organ_utterances)
        if i < len(findings_dict) - 1:
            findings_str += "\n\n"

    return findings_str


def reconstruct_impression(impressions_list):
    if not isinstance(impressions_list, list):
        raise TypeError("impressions_list should be of type list")

    return "\n".join(impressions_list)


def clean_section(text, section="findings:"):
    # Use regex with the section variable to remove the section name, whitespace, and carriage returns before the first character following it
    pattern = rf'\s*{section}\s*\n*\s*(?=\S)'
    cleaned_text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return cleaned_text


def parse_findings(report, do_lower_case=True):
    if do_lower_case:
        report = report.lower()
        valid_organ = [organ.lower() for organ in VALID_ORGANS]
    else:
        valid_organ = VALID_ORGANS

    for section_header in ["findings:", "Findings:"]:
        if section_header in report:
            warnings.warn(
                f"The findings shouldn't start with '{section_header}'. We removed it but this could lead to unexpected behaviors.",
                UserWarning)
            report = clean_section(report, section_header)

    # List of valid organ headers

    # Initialize the result dictionary
    result = {}

    # Split the text into lines
    lines = report.split('\n')

    current_organ = None
    current_utterances = []

    for idx, line in enumerate(lines):
        line_stripped = line.strip()
        if line_stripped == '':
            continue  # Ignore empty lines

        # Remove leading non-word characters (like '-', '*', etc.)
        potential_organ = re.sub(r'^[^\w]*', '', line_stripped)
        # Check if the line ends with a colon
        if potential_organ.endswith(':'):
            potential_organ = potential_organ[:-1].strip()
        else:
            potential_organ = potential_organ.strip()

        if potential_organ in valid_organ:
            # If there's a previous organ, save its utterances
            if current_organ is not None:
                result[current_organ] = current_utterances

            # Start a new organ section
            current_organ = potential_organ + ":"
            current_utterances = []

        elif line_stripped.startswith('-'):
            # It's an utterance line
            if current_organ is not None:
                # Add the utterance to the current organ's list
                utterance = line_stripped
                current_utterances.append(utterance)
            else:
                warnings.warn(f"Utterance without a valid organ header on line {idx + 1}: {line_stripped}")

        else:
            # Line doesn't match any expected pattern
            warnings.warn(f"Unknown organ header on line {idx + 1}: {line_stripped}. Discarding this section.")

            current_organ = None  # Discard this section
            current_utterances = []

    # After looping, save the last organ's utterances if any
    if current_organ is not None:
        result[current_organ] = current_utterances

    # Before returning results, discard empty organs with no utterances
    result = {organ: utterances for organ, utterances in result.items() if utterances}

    return result


def parse_impression(report, do_lower_case=True):
    if do_lower_case:
        report = report.lower()

    for section_header in ["impression:", "Impression:"]:
        if section_header in report:
            warnings.warn(
                f"The impression shouldn't start with '{section_header}'. We removed it but this could lead to unexpected behaviors.",
                UserWarning)
            report = clean_section(report, section_header)

    # Split the report into a list of impressions
    impressions = report.split('\n')
    impressions = [impression.strip() for impression in impressions if impression.strip() != ""]

    # Iterate over the impressions and check numbering
    for index, impression in enumerate(impressions):
        expected_number = index + 1
        # Extract the number at the start
        match = re.match(r'^(\d+)\.', impression.strip())
        if match:
            number = int(match.group(1))

            if number != expected_number:
                warnings.warn(
                    f"Numbering error: Expected {expected_number}, got {number} in impression '{impression}'",
                    UserWarning)
        else:
            warnings.warn(f"Numbering error: Impression does not start with a proper number in '{impression}'",
                          UserWarning)

    return impressions
