from datasets import load_dataset
dialogsum = load_dataset("knkarthick/dialogsum")

text = dialogsum["train"][10014]["summary"]
print(text)
print(dialogsum["train"][10014]["id"])

print("---------------------------------------------")

dialogue = dialogsum["train"][10014]["dialogue"]
print(dialogue)

# choose 3 srcs and 3 tgts for inference
srcs = dialogsum["train"]["dialogue"][:3]
tgts = dialogsum["train"]["summary"][:3]

from psent_score import PSentScorer
psent_scorer.score(srcs, tgts) 

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

PSent_baseline_filtered = psent_scorer.score(dialogue_list, baseline_filtered_list) 

from utils import*

def read_csv_to_df(file):
    import pandas as pd
    df = pd.read_csv(file, encoding='utf-8')
    return df
    
df_ours=read_csv_to_df("./DialSumm_output/Stats_OursPredictions_PEmo.csv")

def compute_correlation_psent(PSent_srcs, PSent_tgts):
    import pandas as pd
    df = pd.DataFrame(
        {'PSent_srcs': PSent_srcs,
         'PSent_tgts': PSent_tgts
         })
    df_selected = df.loc[(df['PSent_srcs'] != 0)]
    # print(df_non_zero)
    # df_selected = df_non_zero[['PSent_srcs', 'PSent_tgts']]
    # print(df_selected)
    r = df_selected.corr(method="spearman")
    r_value = r.iloc[0]['PSent_tgts'] # access the first row
    #                        PSent_srcs  PSent_tgts
    # PSent_srcs       1.000000       0.352869
    # PSent_tgts      0.352869       1.000000
    # ccc = concordance_correlation_coefficient(y_true, y_pred)
    ccc = concordance_correlation_coefficient(df_selected['PSent_srcs'], df_selected['PSent_tgts'])
    # calculate MAE
    # error = mae(actual, calculated)
    error = mae(df_selected['PSent_srcs'], df_selected['PSent_tgts'])
    return {
            "r": r_value,
            "ccc": ccc,
            "error": error
        }

df_our_new = df_ours[['ids', 'PEmoDialogue', 'PEmoPDialogue', 'PEmoNDialogue', 'PEmo_Baseline', 'PEmoP_Baseline', 'PEmoN_Baseline', 'PEmo_BERT_SST_Random', 'PEmoP_BERT_SST_Random', 'PEmoN_BERT_SST_Random', 'PEmo_BERT_DS_SST_Filtration', 'PEmoP_BERT_DS_SST_Filtration', 'PEmoN_BERT_DS_SST_Filtration']]

from sklearn.metrics import mean_absolute_error as mae

compute_correlation_psent(df_our_new["PEmoPDialogue"], df_our_new["PEmoP_Baseline"])

