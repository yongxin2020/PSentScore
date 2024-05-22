# PSentScore
Code for LREC-COLING 2024 paper "PSentScore: Evaluating Sentiment Polarity in Dialogue Summarization"

<--- <img src="./fig/summ_sentiment.png.png" width="300" class="left"><img src="./fig/logo.png" width="400" class="center"> --->
<img src="./fig/summ_sentiment.png" width=650 class="center">

This is the Repo for the paper: [PSentScore: Evaluating Sentiment Polarity in Dialogue Summarization](https://arxiv.org/abs/2307.12371)

## Updates
- 2024.05.17 Release code

## Background
Dialogue Summarization: distilling the most crucial information from human conversations into concise summaries

Mostly focusing on summarizing factual information, neglecting the affective content

Sentiments are however crucial for the analysis of interactive data, especially in healthcare and customer service

## Our Work
We hypothesize that sentiments present in a dialogue should be preserved in its summary, both in terms of proportion (factual vs. affective) and sentiment polarity (global vs. local)

We introduce and assess a set of measures PSentScore aimed at quantifying the preservation of affective content in dialogue summaries

Basic requirements for all the libraries are in the `requirements.txt.`

### Direct use
Our trained BARTScore (on ParaBank2) can be downloaded [here](https://drive.google.com/file/d/1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m/view?usp=sharing). Example usage is shown below.

```python
# To use PSentScore based on BERT-DS-SST3 word-level SA model
>>> from psent_score import PSentScorer
>>> psent_scorer = PSentScorer(["PSent", "PSentP", "PSentN"], checkpoint='/home/getalp/zhouy/Models/bert_ds_sst3_5e-5_5_42')
>>> from datasets import load_dataset
>>> dialogsum = load_dataset("knkarthick/dialogsum")
# srcs: a list of the source texts, tgts: a list of the target texts
# choose 3 srcs and 3 tgts for inference
>>> srcs = dialogsum["train"]["dialogue"][:3]
>>> tgts = dialogsum["train"]["summary"][:3]
>>> psent_scorer.score(srcs, tgts) 
[out]
{'PSent': {'r': 1.0,
  'ccc': 0.17602839621795402,
  'error': 0.029740204118137593},
 'PSentP': {'r': -0.5,
  'ccc': -0.058764523200519794,
  'error': 0.0166242155358542},
 'PSentN': {'r': nan, 'ccc': nan, 'error': 0.018701902364918487}}
```


### Reproduce
To reproduce the results reported in the paper, please see the generated summaries in the folder: `DialSumm_output`, you can use them to conduct analysis.

For analysis, we provide `SUMStat`, `D2TStat` and `WMTStat` in `analysis.py` that can conveniently run analysis. An example of using `SUMStat` is shown below. Detailed usage can refer to `analysis.ipynb`.

```python
>>> from analysis import SUMStat
>>> stat = SUMStat('SUM/REALSumm/final_p.pkl')
>>> stat.evaluate_summary('litepyramid_recall')

[out]
Human metric: litepyramid_recall
metric                                               spearman    kendalltau
-------------------------------------------------  ----------  ------------
rouge1_r                                            0.497526      0.407974
bart_score_cnn_hypo_ref_de_id est                   0.49539       0.392728
bart_score_cnn_hypo_ref_de_Videlicet                0.491011      0.388237
...
```

### Train your custom PSentScore
If you want to train your custom PSentScore with paired data, you can train a sentiment analysis model tailored to your needs. Once you got your trained model (for example, `my_psentscore` folder). You can use your custom PSentScore as shown below.
<-- we provide the scripts and detailed instructions in the `train` folder. -->

```python
>>> from psent_score import PSentScorer
>>> psent_scorer = PSentScorer(["PSent", "PSentP", "PSentN"], checkpoint='my_psentscore') # your custom SA model path
>>> psent_scorer.score(srcs, tgts) # srcs is the list of source texts and tgts is the list of target texts
```


### Notes on use
- Our measure relies on a word-level sentiment analysis model which is biased and might not be available for all languages
- While the BERT-DS-SST3 model demonstrated promising performance on the SST corpus, it has not been evaluated on dialogue corpora

## Bib
Please cite our work if you find it useful.
```
@misc{zhou2024psentscore,
      title={PSentScore: Evaluating Sentiment Polarity in Dialogue Summarization}, 
      author={Yongxin Zhou and Fabien Ringeval and François Portet},
      year={2024},
      eprint={2307.12371},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
