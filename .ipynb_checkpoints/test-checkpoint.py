from psent_score import PSentScorer

psent_scorer = PSentScorer(["PSent", "PSentP", "PSentN"], checkpoint='/home/getalp/zhouy/Models/bert_ds_sst3_5e-5_5_42')

def read_jsonlines_file(file_name):
    import jsonlines
    dialogue_list = []
    summary1_list = []
    summary2_list = []
    summary3_list = []
    ids = []

    with jsonlines.open(file_name) as reader:
        for obj in reader:
            dialogue = obj["dialogue"]
            dialogue_list.append(dialogue)
            summary1 = obj["summary1"] 
            summary1_list.append(summary1)
            summary2 = obj["summary2"]
            summary2_list.append(summary2)
            summary3 = obj["summary3"] 
            summary3_list.append(summary3)
            id = obj["fname"]
            ids.append(id)
    return dialogue_list, summary1_list, summary2_list, summary3_list, ids

import jsonlines
test_file = "./DialSumm_output/dialogsum.test.jsonl" # 500 orws

dialogue_list, summary1_list, summary2_list, summary3_list, ids = read_jsonlines_file(test_file)

def read_text_file(file_name):
    with open(file_name) as f:
        prediction = f.readlines()
    return prediction

Baseline_file = "./DialSumm_output/baseline.txt"
Baseline_filtered_file = "./DialSumm_output/baseline_filtered.txt"

baseline_list = read_text_file(Baseline_file)[:500]
baseline_filtered_list = read_text_file(Baseline_filtered_file)[:500]

PSent_baseline = psent_scorer.score(dialogue_list, baseline_list) 

# {'PSent': {'r': 0.392252163507839, 'ccc': 0.3332693526932844, 'error': 0.02783544588156605}, 'PSentP': {'r': 0.3565232862643794, 'ccc': 0.33816470283593536, 'error': 0.024684006576982543}, 'PSentN': {'r': 0.424709583986352, 'ccc': 0.3408131043523878, 'error': 0.01594073792378572}}

PSent_baseline_filtered = psent_scorer.score(dialogue_list, baseline_filtered_list) 

# {'PSent': {'r': 0.4565470695539414, 'ccc': 0.37487381463717434, 'error': 0.026918887988244527}, 'PSentP': {'r': 0.39043753492088157, 'ccc': 0.38251239414622795, 'error': 0.023132194333878562}, 'PSentN': {'r': 0.49510884209841793, 'ccc': 0.3952607400765182, 'error': 0.015218466996675952}}
