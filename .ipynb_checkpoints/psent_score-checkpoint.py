# %%
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np
from sklearn.metrics import mean_absolute_error as mae
from utils import countPredictionInt, concordance_correlation_coefficient

class PSentScorer: # To use between two sets of data, ex: source dialogues versus. target summaries
    """Calculate rouges scores between two blobs of text.

    Sample usage:
      scorer = PSentScorer(['PSent', 'PSentN'])
      scores = scorer.score(srcs_list,
                            tgts_list)
    """
    def __init__(self, psent_types, checkpoint='/data/yongxin/DiaSumDA/token_classification/ACIInew/bert_ds_sst3/bert_ds_sst3_5e-5_5_42'): # /home/getalp/zhouy/Models/bert_ds_sst3_5e-5_5_42
        # Set up model #  device='cuda:0', 
        self.psent_types = psent_types
        # self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForTokenClassification.from_pretrained(checkpoint)
        #self.model.eval()
        # self.model.to(device)


    def Prediction_sst_model(self, text): # text: a phrase, prediction: positive neutral negative neutral
        inputs = self.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            logits = self.model(**inputs).logits

        predictions = torch.argmax(logits, dim=2)
        predicted_token_class = [self.model.config.id2label[t.item()] for t in predictions[0]]
        # print(predicted_st_tags) # seems not correct
        # print(len(predicted_st_tags))
        # predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]
        # the following lines are for ERC
        #predicted_class_id = logits.argmax().item()
        #prediction = model.config.id2label[predicted_class_id]
        return predicted_token_class

    def sentiment_analysis_PSent(self, text):  # changed from def score_review(review) in lexicon_based_stat.py
        import nltk
        from nltk.tokenize import sent_tokenize
        # sentiment_scores = []
        pos_nb = neg_nb = total_nb = 0
        sents = sent_tokenize(text)
        for sent in sents:  # for each sentence in multiple sentences
            prediction = self.Prediction_sst_model(
                sent)  # prediction: ['neutral', 'neutral', 'neutral', 'neutral', 'neutral'] --> list
            very_negative, negative, neutral, positive, very_positive, total_count = countPredictionInt(
                prediction)  # 0 1 2 1 0 4
            # print(very_negative, negative, neutral, positive, very_positive, total_count)
            pos_nb += very_positive
            pos_nb += positive
            neg_nb += negative
            neg_nb += very_negative
            total_nb += total_count
        # pos_nb, neg_nb, total_nb
        # return sum(sentiment_scores) / len(sentiment_scores)
        proportionSent = (pos_nb + neg_nb) / total_nb
        proportionSentP = pos_nb / total_nb
        proportionSentN = neg_nb / total_nb
        return proportionSent, proportionSentP, proportionSentN


    def CountPSent(self, list):  # changed from def CountEmoP(list) in lexicon_based_stat.py
        PSent = []
        PSentP = []
        PSentN = []

        for i in range(len(list)):  # len(dialogue_list) #100 --> 5 mins
            text = list[i]
            proportionSent, proportionSentP, proportionSentN = self.sentiment_analysis_PSent(text)
            PSent.append(proportionSent)
            PSentP.append(proportionSentP)
            PSentN.append(proportionSentN)

        return PSent, PSentP, PSentN

    
    def compute_correlation_psent(self, PSent_srcs, PSent_tgts):
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

    
    def score(self, srcs, tgts): # modified from df_to_corr(df, metric_type, pre_name) function # srcs and tgts are both list
        #if psent_types is None:
        #    psent_types = ["PSent", "PSentP", "PSentN"]

        PSent_srcs, PSentP_srcs, PSentN_srcs = self.CountPSent(srcs)
        PSent_tgts, PSentP_tgts, PSentN_tgts = self.CountPSent(tgts)

        """Calculates rouge scores between the target and prediction.

        Args:
          srcs: a list of proportion values of the source texts (dialogues)
          tgts: a list of proportion values of the target texts (summaries)
        Returns:
          A dict mapping each psent type to a Score object.
        Raises:
          ValueError: If an invalid psent type is encountered.
        """

        result = {}

        for psent_type in self.psent_types:
            if psent_type == "PSent":
                scores = self.compute_correlation_psent(PSent_srcs, PSent_tgts)
            elif psent_type == "PSentP":
                scores = self.compute_correlation_psent(PSentP_srcs, PSentP_tgts)
            elif psent_type == "PSentN":
                scores = self.compute_correlation_psent(PSentN_srcs, PSentN_tgts)
            else:
                raise ValueError("Invalid psent type: %s" % rouge_type)
            result[psent_type] = scores

        return result


    def test(self):
        """ Test """
        src_list = [
            'This is a very good idea. Although simple, but very insightful.',
            'Can I take a look?',
            'Do not trust him, he is a liar.'
        ]

        tgt_list = [
            "That's stupid.",
            "What's the problem?",
            'He is trustworthy.'
        ]

        print(self.score(src_list, tgt_list))