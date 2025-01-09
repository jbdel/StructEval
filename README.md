```python
from structeval import StructEval

evaluator = StructEval(do_radgraph=True,
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
print(evaluator.forward(refs=refs, hyps=hyps, section="impression", aligned=True))
```