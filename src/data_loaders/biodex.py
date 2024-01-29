from src.utils import extract_labels_from_string, normalize

import os
from collections import Counter, defaultdict
import datasets
import pandas as pd
import json


def _prepare_biodex_dataframe(dataset):
    label = [
        extract_labels_from_string(
            l,
            do_lower=False,
            strip_punct=False,
        )
        for l in dataset["reactions"]
    ]
    df = pd.DataFrame({"text": dataset["fulltext_processed"], "label": label})
    return df


def _load_biodex():
    base_dir = "./data"
    biodex_dir = os.path.join(base_dir, "biodex")

    # get ontology
    biodex_terms = [
        term.strip("\n")
        for term in open(os.path.join(biodex_dir, "reaction_terms.txt")).readlines()
    ]

    # get val and test set
    dataset = datasets.load_dataset("BioDEX/BioDEX-Reactions")
    validation_ds, test_ds = dataset["validation"], dataset["test"]

    # get prior counts, normalized, from the train set
    all_train_reactions = dataset["train"]["reactions"]
    all_train_reactions = [ls.split(", ") for ls in all_train_reactions]
    all_train_reactions = [x for ls in all_train_reactions for x in ls]

    biodex_priors = Counter(all_train_reactions)
    biodex_priors = defaultdict(
        lambda: 0.0,
        {k: v / len(all_train_reactions) for k, v in biodex_priors.items()},
    )
    # save prior
    with open(os.path.join(biodex_dir, "biodex_priors.json"), "w") as fp:
        json.dump(biodex_priors, fp)

    # get correct format df[["text", "label"]]
    validation_df = _prepare_biodex_dataframe(validation_ds)
    test_df = _prepare_biodex_dataframe(test_ds)

    return validation_df, test_df, biodex_terms, None, biodex_priors
