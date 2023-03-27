"""
Script for Transformer Explainability Using
Integrated Gradients with the Captum Package
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import nltk
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from tqdm import trange, tqdm

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from captum.attr import visualization as viz
from captum.attr import LayerConductance, LayerIntegratedGradients, IntegratedGradients

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.metrics import precision_recall_fscore_support as score

nltk.download("stopwords")

from src.models import TransformerClassifier
from src.utils import tokenize


def predict(inputs_ids, attention_mask, tf_idf, tf_transformer):
    
    # If it is a baseline input use 0 tfidf
    if inputs_ids[0][1] == 0:  
        tf_idf = torch.from_numpy(tf_transformer.transform([""]).toarray())

    return model(
        input_ids=inputs_ids,
        attention_mask=attention_mask,
        tfidf_features=tf_idf
    )


def construct_input_ref_pair(text, ref_token_id, sep_token_id, cls_token_id):
    
    number_ids = len(tokenizer.encode(text, add_special_tokens=False))
    encode = tokenizer.encode_plus(
        text,
        max_length=512,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    
    input_ids = encode["input_ids"]
    attention_mask = encode["attention_mask"]

    ref_text = tokenizer.pad_token * number_ids
    ref_input_ids = tokenizer.encode_plus(
        ref_text,
        max_length=512,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )["input_ids"]

    return input_ids, ref_input_ids, number_ids, attention_mask


def construct_input_ref_token_type_pair(input_ids, sep_ind=0):
    seq_len = input_ids.size(1)
    token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]])
    ref_token_type_ids = torch.zeros_like(token_type_ids)  # * -1
    return token_type_ids, ref_token_type_ids


def construct_input_ref_pos_id_pair(input_ids):
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long)
    # we could potentially also use random permutation with `torch.randperm(seq_length, device=device)`
    ref_position_ids = torch.zeros(seq_length, dtype=torch.long)

    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
    return position_ids, ref_position_ids


def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)


def construct_ref_tfidf(ref_input_ids, tfidf):
    seq_len = input_ids.size(1)
    return torch.tensor([[0 for _ in range(seq_len)]])


def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


def process_attribution_token(attr, tokens):
    
    # Get Average of Unique Token in Sentence
    unique_tokens = list(set(tokens))
    mapping = np.array(list(map(list, zip(tokens, attr))))

    output = {}
    for unique in unique_tokens:
        filtered = mapping[mapping[:, 0] == unique]
        # print(filtered)
        get_average = np.mean(filtered[:, 1].astype(float))
        output[unique] = [get_average]

    return pd.DataFrame(output)


if __name__ == "__main__":

    root = ""
    transformer = "microsoft/deberta-v3-base"
    load_model_path = "./deberta-freeze-tfidf.pth"
    save_dir, sample = sys.argv[1], int(sys.argv[2])

    print(save_dir)

    train_csv = "raw_data/augmented_train.csv"

    # ----- Create Result Directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)

	# ----- Load Data

    df_train = pd.read_csv(os.path.join(root, train_csv), header=None)
    df_test = pd.read_csv(os.path.join(root, "raw_data/balancedtest.csv"), header=None)

    df_train.columns = ["cls", "text"]
    df_test.columns = ["cls", "text"]

    # ----- Get TF-IDF

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words=stopwords.words("english"),
        max_df=0.8,
        min_df=10,
        max_features=5096,
    )

    # ----- Get Samples for Each Class in Test Set
    
    df_test_samples = (
        df_test.groupby("cls", as_index=False)
        .apply(lambda x: x.sample(sample))
        .reset_index()[["cls", "text", "level_1"]]
    )

    train_tfidf = tfidf.fit_transform(df_train["text"].tolist())
    test_tfidf = tfidf.transform(df_test_samples["text"].tolist())

    train_tfidf = torch.from_numpy(train_tfidf.toarray())
    test_tfidf = torch.from_numpy(test_tfidf.toarray())

    tfidf_feature_dims = train_tfidf.shape[-1]

    # ----- Load Model & Tokenizer
    
    model = TransformerClassifier(transformer=transformer, tfidf_dim=tfidf_feature_dims)

    state_dict = torch.load(load_model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    # Get reference id
    ref_token_id = tokenizer.pad_token_id  # A token used for generating token reference
    sep_token_id = tokenizer.sep_token_id  # A token used as a separator between question and text and it is also added to the end of the text.
    cls_token_id = tokenizer.cls_token_id  # A token used for prepending to the concatenated question-text word sequence

    attribution_list = []
    predicted_class_labels = []
    true_class_labels = []

    for index in range(len(df_test_samples)):
        
        text, df_true_cls = (
            df_test_samples.iloc[index, :]["text"],
            int(df_test_samples.iloc[index, :]["cls"] - 1),
        )
        
        df_first_tfidf = tfidf.transform([text])
        df_first_tfidf = torch.from_numpy(df_first_tfidf.toarray())

        true_class_labels.append(df_true_cls)

        # Generate Input IDs & Reference IDs
        input_ids, ref_input_ids, sep_id, attention_mask = construct_input_ref_pair(
            text, ref_token_id, sep_token_id, cls_token_id
        )
        token_type_ids, ref_token_type_ids = construct_input_ref_token_type_pair(
            input_ids, sep_id
        )
        position_ids, ref_position_ids = construct_input_ref_pos_id_pair(input_ids)

        indices = input_ids[0].detach().tolist()
        all_tokens = tokenizer.convert_ids_to_tokens(indices)

        # Visualise the Attribution
        eval_output = predict(input_ids, attention_mask, df_first_tfidf, tfidf)
        pred_class = torch.argmax(eval_output[0]).tolist()
        predicted_class_labels.append(pred_class)

        # Integrated Gradient
        lig = LayerIntegratedGradients(predict, model.backbone.embeddings)

        # Target can be Actual Class / Predicted Class
        attributions, delta = lig.attribute(
            inputs=input_ids,
            baselines=ref_input_ids,
            target=pred_class,
            return_convergence_delta=True,
            additional_forward_args=(attention_mask, df_first_tfidf, tfidf),
            n_steps=50,
        )

        attributions_sum = summarize_attributions(attributions)

        # print("Attribution:\n",attributions_sum)

        # Convert to DataFrame
        values = attributions_sum.tolist()
        pd_dict_attr = process_attribution_token(values, all_tokens)

        attribution_list.append(pd_dict_attr)

        score_vis = viz.VisualizationDataRecord(
            word_attributions=attributions_sum,
            pred_prob=torch.softmax(eval_output[0], dim=0)[pred_class],
            pred_class=torch.argmax(eval_output[0]),
            true_class=df_true_cls,
            attr_class=text,
            attr_score=attributions_sum.sum(),
            raw_input_ids=all_tokens,
            convergence_score=delta,
        )
        html = viz.visualize_text([score_vis])

        # save html
        data = html.data
        with open(os.path.join(save_dir, f"sample_{index}.html"), "w") as f:
            f.write(data)

    # ----- Create full DF of all Words & Scores for Tested Samples
    
    df_full = pd.concat(attribution_list, axis=0, ignore_index=True).fillna(0)
    df_full.index = df_test_samples["level_1"].values
    df_full["predicted_class_labels"] = predicted_class_labels
    df_full["true_class_labels"] = true_class_labels
    df_full.to_csv(os.path.join(save_dir, "attribution_scores.csv"))