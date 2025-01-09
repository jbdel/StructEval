import pytest
import json
from structeval.StructEval import StructEval
from structeval.utils import remove_bullets, remove_numbering, reconstruct_findings, reconstruct_impression
from itertools import chain


def test_remove_bullets():
    findings = ["- Finding one", "- Finding two", "No bullet"]
    expected = ["Finding one", "Finding two", "No bullet"]
    assert remove_bullets(findings) == expected


def test_remove_numbering():
    impressions = ["1. Impression one", "2. Impression two", "No number"]
    expected = ["Impression one", "Impression two", "No number"]
    assert remove_numbering(impressions) == expected


def test_reconstruct_findings():
    findings_dict = {
        "Organ A:": ["Finding A1", "Finding A2"],
        "Organ B:": ["Finding B1"]
    }
    expected = "Organ A:\nFinding A1\nFinding A2\n\nOrgan B:\nFinding B1"
    assert reconstruct_findings(findings_dict) == expected


def test_reconstruct_impression():
    impressions_list = ["Impression one", "Impression two"]
    expected = "Impression one\nImpression two"
    assert reconstruct_impression(impressions_list) == expected


# def test_parse_and_reconstruct_findings():
#     evaluator = StructEval()
#     all_utterances = []
#
#     # Load the JSON file for findings
#     try:
#         with open("collated_studies_closed_headings_findings5.json", "r") as f:
#             reports = json.load(f)
#     except FileNotFoundError:
#         pytest.skip("File 'collated_studies_closed_headings_findings5.json' not found.")
#
#     for k, v in reports.items():
#         findings = v["findings_section"]
#         findings_dict = evaluator.parse_findings(findings, do_lower_case=False)
#         findings_section = reconstruct_findings(findings_dict)
#         assert findings_section == findings, f"Findings reconstruction mismatch for report {k}"
#
#         impression = v.get("impression_section", "")
#         if impression:
#             impression_list = evaluator.parse_impression(impression, do_lower_case=False)
#             impression_section = reconstruct_impression(impression_list)
#             assert impression_section == impression, f"Impression reconstruction mismatch for report {k}"
#
#             utterances_impression = remove_numbering(impression_list)
#             all_utterances.extend(utterances_impression)
#
#         utterances_findings = remove_bullets(list(chain.from_iterable(findings_dict.values())))
#         all_utterances.extend(utterances_findings)
#
#     # Load the JSON file for impressions
#     try:
#         with open("collated_studies_impression4.json", "r") as f:
#             reports = json.load(f)
#     except FileNotFoundError:
#         pytest.skip("File 'collated_studies_impression4.json' not found.")
#
#     for k, v in reports.items():
#         impression = v["impression_section"]
#         impression_list = evaluator.parse_impression(impression, do_lower_case=False)
#         impression_section = reconstruct_impression(impression_list)
#         assert impression_section == impression, f"Impression reconstruction mismatch for report {k}"
#
#         utterances_impression = remove_numbering(impression_list)
#         all_utterances.extend(utterances_impression)
#
#     # Optionally, you can perform further assertions or checks on all_utterances
#     assert len(all_utterances) > 0, "No utterances were extracted."


def test_not_structured():
    # Sample references and hypotheses
    refs = [
        "No acute cardiopulmonary process.",
        "No radiographic findings to suggest pneumonia.",
        "1.Status post median sternotomy for CABG with stable cardiac enlargement and calcification of the aorta consistent with atherosclerosis.Relatively lower lung volumes with no focal airspace consolidation appreciated.Crowding of the pulmonary vasculature with possible minimal perihilar edema, but no overt pulmonary edema.No pleural effusions or pneumothoraces.",
        "1. Left PICC tip appears to terminate in the distal left brachiocephalic vein.2. Mild pulmonary vascular congestion.3. Interval improvement in aeration of the lung bases with residual streaky opacity likely reflective of atelectasis.Interval resolution of the left pleural effusion.",
        "No definite acute cardiopulmonary process.Enlarged cardiac silhouette could be accentuated by patient's positioning.",
        "Increased mild pulmonary edema and left basal atelectasis.",
    ]

    hyps = [
        "No acute cardiopulmonary process.",
        "No radiographic findings to suggest pneumonia.",
        "Status post median sternotomy for CABG with stable cardiac enlargement and calcification of the aorta consistent with atherosclerosis.",
        "Relatively lower lung volumes with no focal airspace consolidation appreciated.",
        "Crowding of the pulmonary vasculature with possible minimal perihilar edema, but no overt pulmonary edema.",
        "No pleural effusions or pneumothoraces.",
    ]

    # Instantiate RadEval with desired configurations
    evaluator = StructEval(do_radgraph=True,
                           do_green=False,
                           do_bleu=True,
                           do_rouge=True,
                           do_bertscore=True,
                           do_diseases=False)

    # Compute scores
    results = evaluator(refs=refs, hyps=hyps, aligned=False)

    # Expected result with pytest.approx for approximate comparison
    expected_result = {
        "radgraph_simple": pytest.approx(0.41111111111111115, 0.01),
        "radgraph_partial": pytest.approx(0.41111111111111115, 0.01),
        "radgraph_complete": pytest.approx(0.41414141414141414, 0.01),
        "bleu": pytest.approx(0.16681006823938177, 0.01),
        "bertscore": pytest.approx(0.6327474117279053, 0.01),
        "rouge1": pytest.approx(0.44681719607092746, 0.01),
        "rouge2": pytest.approx(0.4205128205128205, 0.01),
        "rougeL": pytest.approx(0.44681719607092746, 0.01),
        # "chexbert-5_micro avg_f1-score": pytest.approx(0.2857142857142857, 0.01),
        # "chexbert-all_micro avg_f1-score": pytest.approx(0.3333333333333333, 0.01),
        # "chexbert-5_macro avg_f1-score": pytest.approx(0.13333333333333333, 0.01),
        # "chexbert-all_macro avg_f1-score": pytest.approx(0.08333333333333333, 0.01),
    }

    # Compare computed results with expected results
    for key, expected_value in expected_result.items():
        assert key in results, f"Missing key in results: {key}"
        assert results[key] == expected_value, f"Mismatch for {key}: {results[key]} != {expected_value}"


def test_struxt_eval_impression_aligned():
    rr = StructEval(do_radgraph=True,
                    do_green=False,
                    do_bleu=True,
                    do_rouge=True,
                    do_bertscore=True,
                    do_diseases=True)

    hyps = [
        '1. No evidence of pneumothorax following the procedure.\n2. No significant change in the appearance of the chest when compared to the previous study.',
        '1. Improvement in vascular congestion in the upper lobes.\n2. Unchanged bibasilar consolidation, left greater than right, likely representing a combination of atelectasis and residual dependent edema.\n3. Small bilateral pleural effusions are present.\n4. Possible pneumonia in the left lower lobe.\n5. Mild-to-moderate enlargement of the cardiac silhouette, which may suggest cardiomegaly and/or pericardial effusion.',
        '1. Improved inspiratory effort compared to the previous study.\n2. Persistent enlargement of the cardiac silhouette with pacemaker lead extending to the apex of the right ventricle.\n3. No radiographic evidence of acute pneumonia or vascular congestion.',
        '1. The tracheostomy tube is well-positioned without evidence of pneumothorax or pneumomediastinum.\n2. No significant change in the appearance of the heart and lungs when compared to the previous study.',
        '1. Endotracheal tube tip is positioned 3.7 cm above the carina.\n2. Nasogastric tube tip is appropriately located in the stomach.\n3. Stable heart size and mediastinal contours.\n4. Increased left pleural effusion.\n5. Worsening of left retrocardiac consolidation.',
        '1. Improved ventilation of the postoperative right lung.\n2. Expected appearance of the right mediastinal border post-esophagectomy.\n3. Unchanged position of monitoring and support devices.\n4. Slight decrease in soft tissue air collection.\n5. Minimally increased retrocardiac atelectasis.\n6. Normal left lung.',
    ]

    refs = [
        '1. No evidence of pneumothorax following the procedure.\n2. No significant change in the appearance of the chest when compared to the previous study.',
        '1. Improvement in vascular congestion in the upper lobes.\n2. Unchanged bibasilar consolidation, left greater than right, likely representing a combination of atelectasis and residual dependent edema.\n3. Small bilateral pleural effusions are present.\n4. Possible pneumonia in the left lower lobe.\n5. Mild-to-moderate enlargement of the cardiac silhouette, which may suggest cardiomegaly and/or pericardial effusion.',
        '1. Improved inspiratory effort compared to the previous study.\n2. Persistent enlargement of the cardiac silhouette with pacemaker lead extending to the apex of the right ventricle.\n3. No radiographic evidence of acute pneumonia or vascular congestion.',
        '1. No evidence of pneumothorax following the procedure.\n2. No significant change in the appearance of the chest when compared to the previous study.',
        '1. Improvement in vascular congestion in the upper lobes.\n2. Unchanged bibasilar consolidation, left greater than right, likely representing a combination of atelectasis and residual dependent edema.\n3. Small bilateral pleural effusions are present.\n4. Possible pneumonia in the left lower lobe.\n5. Mild-to-moderate enlargement of the cardiac silhouette, which may suggest cardiomegaly and/or pericardial effusion.',
        '1. Improved inspiratory effort compared to the previous study.\n2. Persistent enlargement of the cardiac silhouette with pacemaker lead extending to the apex of the right ventricle.\n3. No radiographic evidence of acute pneumonia or vascular congestion.',
    ]

    result = rr.forward(refs=refs, hyps=hyps, section="impression", aligned=True)
    expected_result = {
        'radgraph_simple': pytest.approx(0.611111111111111, 0.01),
        'radgraph_partial': pytest.approx(0.611111111111111, 0.01),
        'radgraph_complete': pytest.approx(0.5484848484848485, 0.01),
        'bleu': pytest.approx(0.5805253039653325, 0.01),
        'bertscore': pytest.approx(0.7236699461936951, 0.01),
        'rouge1': pytest.approx(0.6553281392097181, 0.01),
        'rouge2': pytest.approx(0.5955337690631808, 0.01),
        'rougeL': pytest.approx(0.6465562093851568, 0.01),
        'samples_avg_precision': pytest.approx(0.638888888888889, 0.01),
        'samples_avg_recall': pytest.approx(0.638888888888889, 0.01),
        'samples_avg_f1-score': pytest.approx(0.638888888888889, 0.01)
    }
    for key in expected_result:
        assert result[key] == expected_result[key], f"{key} does not match expected value"


def test_struxt_eval_impression_not_aligned():
    rr = StructEval(do_radgraph=True,
                    do_green=False,
                    do_bleu=True,
                    do_rouge=True,
                    do_bertscore=True,
                    do_diseases=True)

    hyps = [
        '1. No evidence of pneumothorax following the procedure.\n2. No significant change in the appearance of the chest when compared to the previous study.',
        '1. Improvement in vascular congestion in the upper lobes.\n2. Unchanged bibasilar consolidation, left greater than right, likely representing a combination of atelectasis and residual dependent edema.\n3. Small bilateral pleural effusions are present.\n4. Possible pneumonia in the left lower lobe.\n5. Mild-to-moderate enlargement of the cardiac silhouette, which may suggest cardiomegaly and/or pericardial effusion.',
        '1. Improved inspiratory effort compared to the previous study.\n2. Persistent enlargement of the cardiac silhouette with pacemaker lead extending to the apex of the right ventricle.\n3. No radiographic evidence of acute pneumonia or vascular congestion.',
        '1. The tracheostomy tube is well-positioned without evidence of pneumothorax or pneumomediastinum.\n2. No significant change in the appearance of the heart and lungs when compared to the previous study.',
        '1. Endotracheal tube tip is positioned 3.7 cm above the carina.\n2. Nasogastric tube tip is appropriately located in the stomach.\n3. Stable heart size and mediastinal contours.\n4. Increased left pleural effusion.\n5. Worsening of left retrocardiac consolidation.',
        '1. Improved ventilation of the postoperative right lung.\n2. Expected appearance of the right mediastinal border post-esophagectomy.\n3. Unchanged position of monitoring and support devices.\n4. Slight decrease in soft tissue air collection.\n5. Minimally increased retrocardiac atelectasis.\n6. Normal left lung.',
    ]

    refs = [
        '1. No evidence of pneumothorax following the procedure.\n2. No significant change in the appearance of the chest when compared to the previous study.',
        '1. Improvement in vascular congestion in the upper lobes.\n2. Unchanged bibasilar consolidation, left greater than right, likely representing a combination of atelectasis and residual dependent edema.\n3. Small bilateral pleural effusions are present.\n4. Possible pneumonia in the left lower lobe.\n5. Mild-to-moderate enlargement of the cardiac silhouette, which may suggest cardiomegaly and/or pericardial effusion.',
        '1. Improved inspiratory effort compared to the previous study.\n2. Persistent enlargement of the cardiac silhouette with pacemaker lead extending to the apex of the right ventricle.\n3. No radiographic evidence of acute pneumonia or vascular congestion.',
        '1. No evidence of pneumothorax following the procedure.\n2. No significant change in the appearance of the chest when compared to the previous study.',
        '1. Improvement in vascular congestion in the upper lobes.\n2. Unchanged bibasilar consolidation, left greater than right, likely representing a combination of atelectasis and residual dependent edema.\n3. Small bilateral pleural effusions are present.\n4. Possible pneumonia in the left lower lobe.\n5. Mild-to-moderate enlargement of the cardiac silhouette, which may suggest cardiomegaly and/or pericardial effusion.',
        '1. Improved inspiratory effort compared to the previous study.\n2. Persistent enlargement of the cardiac silhouette with pacemaker lead extending to the apex of the right ventricle.\n3. No radiographic evidence of acute pneumonia or vascular congestion.',
    ]

    result = rr.forward(refs=refs, hyps=hyps, section="impression", aligned=False)
    expected_result = {
        'radgraph_simple': pytest.approx(0.6095619658119659, 0.01),
        'radgraph_partial': pytest.approx(0.6090788740245262, 0.01),
        'radgraph_complete': pytest.approx(0.5283950617283951, 0.01),
        'bleu': pytest.approx(0.5909341152594585, 0.01),
        'bertscore': pytest.approx(0.7889277338981628, 0.01),
        'rouge1': pytest.approx(0.7133838383838383, 0.01),
        'rouge2': pytest.approx(0.6183064919692035, 0.01),
        'rougeL': pytest.approx(0.6795875420875421, 0.01),
        'samples_avg_precision': pytest.approx(0.8333333333333334, 0.01),
        'samples_avg_recall': pytest.approx(0.6642857142857143, 0.01),
        'samples_avg_f1-score': pytest.approx(0.7194444444444444, 0.01)
    }
    for key in expected_result:
        assert result[key] == expected_result[key], f"{key} does not match expected value"


def test_struxt_eval_findings_aligned():
    rr = StructEval(do_radgraph=True,
                    do_green=False,
                    do_bleu=True,
                    do_rouge=True,
                    do_bertscore=True,
                    do_diseases=True)

    refs = [
        'Lungs and Airways:\n- Distortion of the pulmonary bronchovascular markings suggestive of COPD.\n- Lung volumes are within normal limits.\n- No consolidation or pneumothorax observed.\n- Minimal atelectasis at the left lung base.\n\nTubes, Catheters, and Support Devices:\n- Endotracheal tube in situ, terminating 3 cm above the carina.\n- Nasoenteric tube in situ, tip below the left hemidiaphragm, not visualized on this radiograph.\n\nPleura:\n- No pleural effusion seen.',
        'Lungs and Airways:\n- Bilateral diffuse fluffy opacities, suggestive of pulmonary edema.\n\nPleura:\n- Loss of visualization of the bilateral hemidiaphragms, suggesting layering effusions.\n\nCardiovascular:\n- Stable cardiomegaly.\n\nMusculoskeletal and Chest Wall:\n- Chronic impacted fracture of the left humeral surgical neck with periosteal new bone formation.\n- Dislocation of the humerus from the glenoid.',
        'Lungs and Airways:\n- The lungs are clear.\n\nPleura:\n- No pleural effusion or pneumothorax identified.\n\nCardiovascular:\n- Mild pulmonary vascular engorgement noted, but no interstitial edema.\n- Cardiomediastinal silhouette is stable.\n\nTubes, Catheters, and Support Devices:\n- Inferior approach hemodialysis catheter terminating in the right atrium.\n\nMusculoskeletal and Chest Wall:\n- Patient is rotated to the right.',
        'Lungs and Airways:\n- No focal consolidation, effusion, edema, or pneumothorax.\n- Minimal left basilar atelectasis.\n\nCardiovascular:\n- The heart is normal in size.\n\nHila and Mediastinum:\n- Fullness of the left hilum appears unchanged.\n- The descending thoracic aorta is tortuous.',
        'Lungs and Airways:\n- Low lung volumes.\n- No definite focal consolidation.\n- Streaky opacities suggesting atelectasis.\n\nPleura:\n- No pleural effusion.\n- No pneumothorax.\n\nCardiovascular:\n- Unremarkable cardiac silhouette.',
        'Lungs and Airways:\n- Interval resolution of the opacity in the anterior segment of the upper lobe.\n- Subtle persistent opacity at the right lung base laterally, potentially within the right lower lobe.\n\nCardiovascular:\n- Enlarged cardiac silhouette, stable in appearance.\n\nPleura:\n- Posterior costophrenic angles are sharp.\n\nMusculoskeletal and Chest Wall:\n- Osseous and soft tissue structures are unremarkable.',
        'Tubes, Catheters, and Support Devices:\n- Left-sided Automatic Implantable Cardioverter-Defibrillator (AICD) in place\n- Swan Ganz catheter terminating in the right descending pulmonary artery\n- Sternotomy wires intact and aligned\n- Intra-aortic balloon pump previously present has been removed\n\nLungs and Airways:\n- No evidence of pneumothorax\n- Lungs are clear\n\nCardiovascular:\n- Moderate cardiomegaly, stable',
        'Lungs and Airways:\n- The lungs are clear.\n\nCardiovascular:\n- The cardiomediastinal silhouette is within normal limits.\n\nMusculoskeletal and Chest Wall:\n- No acute osseous abnormalities.',
        'Lungs and Airways:\n- The lungs are clear.\n\nPleura:\n- No effusion or pneumothorax is present.\n\nCardiovascular:\n- The cardiomediastinal silhouette is normal.\n\nMusculoskeletal and Chest Wall:\n- Osseous structures and soft tissues are unremarkable.',
        'Cardiovascular:\n- Moderate cardiomegaly.\n\nLungs and Airways:\n- Hyperinflated lungs.\n- Biapical scarring without change.\n\nPleura:\n- No pneumothorax or enlarging pleural effusion.\n- Chronic blunting of the right costophrenic angle, which may represent a small effusion or scarring.\n\nMusculoskeletal and Chest Wall:\n- Moderate degenerative changes in the thoracic spine.'
    ]
    hyps = [
        'Lungs and Airways:\n- Distortion of the pulmonary bronchovascular markings suggestive of COPD.\n- Lung volumes are within normal limits.\n- No consolidation or pneumothorax observed.\n- Minimal atelectasis at the left lung base.\n\nTubes, Catheters, and Support Devices:\n- Endotracheal tube in situ, terminating 3 cm above the carina.\n- Nasoenteric tube in situ, tip below the left hemidiaphragm, not visualized on this radiograph.\n\nPleura:\n- No pleural effusion seen.',
        'Lungs and Airways:\n- Bilateral diffuse fluffy opacities, suggestive of pulmonary edema.\n\nPleura:\n- Loss of visualization of the bilateral hemidiaphragms, suggesting layering effusions.\n\nCardiovascular:\n- Stable cardiomegaly.\n\nMusculoskeletal and Chest Wall:\n- Chronic impacted fracture of the left humeral surgical neck with periosteal new bone formation.\n- Dislocation of the humerus from the glenoid.',
        'Lungs and Airways:\n- The lungs are clear.\n\nPleura:\n- No pleural effusion or pneumothorax identified.\n\nCardiovascular:\n- Mild pulmonary vascular engorgement noted, but no interstitial edema.\n- Cardiomediastinal silhouette is stable.\n\nTubes, Catheters, and Support Devices:\n- Inferior approach hemodialysis catheter terminating in the right atrium.\n\nMusculoskeletal and Chest Wall:\n- Patient is rotated to the right.',
        'Lungs and Airways:\n- No focal consolidation, effusion, edema, or pneumothorax.\n- Minimal left basilar atelectasis.\n\nCardiovascular:\n- The heart is normal in size.\n\nHila and Mediastinum:\n- Fullness of the left hilum appears unchanged.\n- The descending thoracic aorta is tortuous.',
        'Lungs and Airways:\n- Low lung volumes.\n- No definite focal consolidation.\n- Streaky opacities suggesting atelectasis.\n\nPleura:\n- No pleural effusion.\n- No pneumothorax.\n\nCardiovascular:\n- Unremarkable cardiac silhouette.',
        'Lungs and Airways:\n- Bilateral diffuse fluffy opacities, suggestive of pulmonary edema.\n\nPleura:\n- Loss of visualization of the bilateral hemidiaphragms, suggesting layering effusions.\n\nCardiovascular:\n- Stable cardiomegaly.\n\nMusculoskeletal and Chest Wall:\n- Chronic impacted fracture of the left humeral surgical neck with periosteal new bone formation.\n- Dislocation of the humerus from the glenoid.',
        'Lungs and Airways:\n- The lungs are clear.\n\nPleura:\n- No pleural effusion or pneumothorax identified.\n\nCardiovascular:\n- Mild pulmonary vascular engorgement noted, but no interstitial edema.\n- Cardiomediastinal silhouette is stable.\n\nTubes, Catheters, and Support Devices:\n- Inferior approach hemodialysis catheter terminating in the right atrium.\n\nMusculoskeletal and Chest Wall:\n- Patient is rotated to the right.',
        'Lungs and Airways:\n- No focal consolidation, effusion, edema, or pneumothorax.\n- Minimal left basilar atelectasis.\n\nCardiovascular:\n- The heart is normal in size.\n\nHila and Mediastinum:\n- Fullness of the left hilum appears unchanged.\n- The descending thoracic aorta is tortuous.',
        'Lungs and Airways:\n- Low lung volumes.\n- No definite focal consolidation.\n- Streaky opacities suggesting atelectasis.\n\nPleura:\n- No pleural effusion.\n- No pneumothorax.\n\nCardiovascular:\n- Unremarkable cardiac silhouette.',
        'Lungs and Airways:\n- Distortion of the pulmonary bronchovascular markings suggestive of COPD.\n- Lung volumes are within normal limits.\n- No consolidation or pneumothorax observed.\n- Minimal atelectasis at the left lung base.\n\nTubes, Catheters, and Support Devices:\n- Endotracheal tube in situ, terminating 3 cm above the carina.\n- Nasoenteric tube in situ, tip below the left hemidiaphragm, not visualized on this radiograph.\n\nPleura:\n- No pleural effusion seen.',
    ]

    # Assuming all refs and hyps are included as in main3
    results = rr.forward(refs=refs, hyps=hyps, section="findings", aligned=True)

    expected_overall_avg_scores = {
        'radgraph_simple': pytest.approx(0.4604166666666666, 0.01),
        'radgraph_partial': pytest.approx(0.4395833333333333, 0.01),
        'radgraph_complete': pytest.approx(0.4333333333333333, 0.01),
        'bleu': pytest.approx(0.3766061387175529, 0.01),
        'bertscore': pytest.approx(0.539288105815649, 0.01),
        'rouge1': pytest.approx(0.5028678266178266, 0.01),
        'rouge2': pytest.approx(0.453125, 0.01),
        'rougeL': pytest.approx(0.5028678266178266, 0.01),
        'samples_avg_precision': pytest.approx(0.65625, 0.01),
        'samples_avg_recall': pytest.approx(0.6375, 0.01),
        'samples_avg_f1-score': pytest.approx(0.64375, 0.01)
    }

    for key in expected_overall_avg_scores:
        assert results["overall_avg_scores"][key] == expected_overall_avg_scores[
            key], f"{key} does not match expected value"

    expected_section_scores = {
        'section_avg_precision': pytest.approx(0.8933333333333332, 0.01),
        'section_avg_recall': pytest.approx(0.8916666666666666, 0.01),
        'section_avg_f1-score': pytest.approx(0.8845238095238095, 0.01)
    }

    for key in expected_section_scores:
        assert results["section_scores"][key] == expected_section_scores[key], f"{key} does not match expected value"


def test_struxt_eval_findings_not_aligned():
    rr = StructEval(do_radgraph=True,
                    do_green=False,
                    do_bleu=True,
                    do_rouge=True,
                    do_bertscore=True,
                    do_diseases=True)

    refs = [
        'Lungs and Airways:\n- Distortion of the pulmonary bronchovascular markings suggestive of COPD.\n- Lung volumes are within normal limits.\n- No consolidation or pneumothorax observed.\n- Minimal atelectasis at the left lung base.\n\nTubes, Catheters, and Support Devices:\n- Endotracheal tube in situ, terminating 3 cm above the carina.\n- Nasoenteric tube in situ, tip below the left hemidiaphragm, not visualized on this radiograph.\n\nPleura:\n- No pleural effusion seen.',
        'Lungs and Airways:\n- Bilateral diffuse fluffy opacities, suggestive of pulmonary edema.\n\nPleura:\n- Loss of visualization of the bilateral hemidiaphragms, suggesting layering effusions.\n\nCardiovascular:\n- Stable cardiomegaly.\n\nMusculoskeletal and Chest Wall:\n- Chronic impacted fracture of the left humeral surgical neck with periosteal new bone formation.\n- Dislocation of the humerus from the glenoid.',
        'Lungs and Airways:\n- The lungs are clear.\n\nPleura:\n- No pleural effusion or pneumothorax identified.\n\nCardiovascular:\n- Mild pulmonary vascular engorgement noted, but no interstitial edema.\n- Cardiomediastinal silhouette is stable.\n\nTubes, Catheters, and Support Devices:\n- Inferior approach hemodialysis catheter terminating in the right atrium.\n\nMusculoskeletal and Chest Wall:\n- Patient is rotated to the right.',
        'Lungs and Airways:\n- No focal consolidation, effusion, edema, or pneumothorax.\n- Minimal left basilar atelectasis.\n\nCardiovascular:\n- The heart is normal in size.\n\nHila and Mediastinum:\n- Fullness of the left hilum appears unchanged.\n- The descending thoracic aorta is tortuous.',
        'Lungs and Airways:\n- Low lung volumes.\n- No definite focal consolidation.\n- Streaky opacities suggesting atelectasis.\n\nPleura:\n- No pleural effusion.\n- No pneumothorax.\n\nCardiovascular:\n- Unremarkable cardiac silhouette.',
        'Lungs and Airways:\n- Interval resolution of the opacity in the anterior segment of the upper lobe.\n- Subtle persistent opacity at the right lung base laterally, potentially within the right lower lobe.\n\nCardiovascular:\n- Enlarged cardiac silhouette, stable in appearance.\n\nPleura:\n- Posterior costophrenic angles are sharp.\n\nMusculoskeletal and Chest Wall:\n- Osseous and soft tissue structures are unremarkable.',
        'Tubes, Catheters, and Support Devices:\n- Left-sided Automatic Implantable Cardioverter-Defibrillator (AICD) in place\n- Swan Ganz catheter terminating in the right descending pulmonary artery\n- Sternotomy wires intact and aligned\n- Intra-aortic balloon pump previously present has been removed\n\nLungs and Airways:\n- No evidence of pneumothorax\n- Lungs are clear\n\nCardiovascular:\n- Moderate cardiomegaly, stable',
        'Lungs and Airways:\n- The lungs are clear.\n\nCardiovascular:\n- The cardiomediastinal silhouette is within normal limits.\n\nMusculoskeletal and Chest Wall:\n- No acute osseous abnormalities.',
        'Lungs and Airways:\n- The lungs are clear.\n\nPleura:\n- No effusion or pneumothorax is present.\n\nCardiovascular:\n- The cardiomediastinal silhouette is normal.\n\nMusculoskeletal and Chest Wall:\n- Osseous structures and soft tissues are unremarkable.',
        'Cardiovascular:\n- Moderate cardiomegaly.\n\nLungs and Airways:\n- Hyperinflated lungs.\n- Biapical scarring without change.\n\nPleura:\n- No pneumothorax or enlarging pleural effusion.\n- Chronic blunting of the right costophrenic angle, which may represent a small effusion or scarring.\n\nMusculoskeletal and Chest Wall:\n- Moderate degenerative changes in the thoracic spine.'
    ]
    hyps = [
        'Lungs and Airways:\n- Distortion of the pulmonary bronchovascular markings suggestive of COPD.\n- Lung volumes are within normal limits.\n- No consolidation or pneumothorax observed.\n- Minimal atelectasis at the left lung base.\n\nTubes, Catheters, and Support Devices:\n- Endotracheal tube in situ, terminating 3 cm above the carina.\n- Nasoenteric tube in situ, tip below the left hemidiaphragm, not visualized on this radiograph.\n\nPleura:\n- No pleural effusion seen.',
        'Lungs and Airways:\n- Bilateral diffuse fluffy opacities, suggestive of pulmonary edema.\n\nPleura:\n- Loss of visualization of the bilateral hemidiaphragms, suggesting layering effusions.\n\nCardiovascular:\n- Stable cardiomegaly.\n\nMusculoskeletal and Chest Wall:\n- Chronic impacted fracture of the left humeral surgical neck with periosteal new bone formation.\n- Dislocation of the humerus from the glenoid.',
        'Lungs and Airways:\n- The lungs are clear.\n\nPleura:\n- No pleural effusion or pneumothorax identified.\n\nCardiovascular:\n- Mild pulmonary vascular engorgement noted, but no interstitial edema.\n- Cardiomediastinal silhouette is stable.\n\nTubes, Catheters, and Support Devices:\n- Inferior approach hemodialysis catheter terminating in the right atrium.\n\nMusculoskeletal and Chest Wall:\n- Patient is rotated to the right.',
        'Lungs and Airways:\n- No focal consolidation, effusion, edema, or pneumothorax.\n- Minimal left basilar atelectasis.\n\nCardiovascular:\n- The heart is normal in size.\n\nHila and Mediastinum:\n- Fullness of the left hilum appears unchanged.\n- The descending thoracic aorta is tortuous.',
        'Lungs and Airways:\n- Low lung volumes.\n- No definite focal consolidation.\n- Streaky opacities suggesting atelectasis.\n\nPleura:\n- No pleural effusion.\n- No pneumothorax.\n\nCardiovascular:\n- Unremarkable cardiac silhouette.',
        'Lungs and Airways:\n- Bilateral diffuse fluffy opacities, suggestive of pulmonary edema.\n\nPleura:\n- Loss of visualization of the bilateral hemidiaphragms, suggesting layering effusions.\n\nCardiovascular:\n- Stable cardiomegaly.\n\nMusculoskeletal and Chest Wall:\n- Chronic impacted fracture of the left humeral surgical neck with periosteal new bone formation.\n- Dislocation of the humerus from the glenoid.',
        'Lungs and Airways:\n- The lungs are clear.\n\nPleura:\n- No pleural effusion or pneumothorax identified.\n\nCardiovascular:\n- Mild pulmonary vascular engorgement noted, but no interstitial edema.\n- Cardiomediastinal silhouette is stable.\n\nTubes, Catheters, and Support Devices:\n- Inferior approach hemodialysis catheter terminating in the right atrium.\n\nMusculoskeletal and Chest Wall:\n- Patient is rotated to the right.',
        'Lungs and Airways:\n- No focal consolidation, effusion, edema, or pneumothorax.\n- Minimal left basilar atelectasis.\n\nCardiovascular:\n- The heart is normal in size.\n\nHila and Mediastinum:\n- Fullness of the left hilum appears unchanged.\n- The descending thoracic aorta is tortuous.',
        'Lungs and Airways:\n- Low lung volumes.\n- No definite focal consolidation.\n- Streaky opacities suggesting atelectasis.\n\nPleura:\n- No pleural effusion.\n- No pneumothorax.\n\nCardiovascular:\n- Unremarkable cardiac silhouette.',
        'Lungs and Airways:\n- Distortion of the pulmonary bronchovascular markings suggestive of COPD.\n- Lung volumes are within normal limits.\n- No consolidation or pneumothorax observed.\n- Minimal atelectasis at the left lung base.\n\nTubes, Catheters, and Support Devices:\n- Endotracheal tube in situ, terminating 3 cm above the carina.\n- Nasoenteric tube in situ, tip below the left hemidiaphragm, not visualized on this radiograph.\n\nPleura:\n- No pleural effusion seen.',
    ]

    results = rr.forward(refs=refs, hyps=hyps, section="findings", aligned=False)

    expected_overall_avg_scores = {
        'radgraph_simple': pytest.approx(0.4844642857142857, 0.01),
        'radgraph_partial': pytest.approx(0.4625595238095238, 0.01),
        'radgraph_complete': pytest.approx(0.45125000000000004, 0.01),
        'bleu': pytest.approx(0.4018975912673429, 0.01),
        'bertscore': pytest.approx(0.5748156843706965, 0.01),
        'rouge1': pytest.approx(0.5240148323898323, 0.01),
        'rouge2': pytest.approx(0.4682850241545894, 0.01),
        'rougeL': pytest.approx(0.5240148323898323, 0.01),
        'samples_avg_precision': pytest.approx(0.6583333333333333, 0.01),
        'samples_avg_recall': pytest.approx(0.6458333333333333, 0.01),
        'samples_avg_f1-score': pytest.approx(0.6391666666666667, 0.01)
    }

    for key in expected_overall_avg_scores:
        assert results["overall_avg_scores"][key] == expected_overall_avg_scores[
            key], f"{key} does not match expected value"

    expected_section_scores = {
        'section_avg_precision': pytest.approx(0.8933333333333332, 0.01),
        'section_avg_recall': pytest.approx(0.8916666666666666, 0.01),
        'section_avg_f1-score': pytest.approx(0.8845238095238095, 0.01)
    }

    for key in expected_section_scores:
        assert results["section_scores"][key] == expected_section_scores[key], f"{key} does not match expected value"
