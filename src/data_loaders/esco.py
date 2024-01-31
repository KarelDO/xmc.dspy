import os
from collections import defaultdict
import pandas as pd
import json

def _load_esco_ontology(esco_dir):
    # read file
    ontology_file = os.path.join(esco_dir, "skills_en_label.csv")
    ontology_prior = os.path.join(esco_dir, "esco_priors.json")

    # get skills and descriptions
    ontology = pd.read_csv(ontology_file)
    skills_and_description = ontology[["preferredLabel", "description"]].to_numpy()
    skills = [x[0] for x in skills_and_description]
    descriptions = [x[1] for x in skills_and_description]

    # get priors
    with open(ontology_prior, "r") as f:
        priors = defaultdict(lambda: 0.0)
        priors.update(json.load(f))

    return skills, descriptions, priors


def _prepare_esco_dataframe(path):
    df = pd.read_csv(path)
    df = df.rename(columns={"sentence": "text"})

    # filter unusable rows
    df = df[df["label"] != "LABEL NOT PRESENT"]
    df = df[df["label"] != "UNDERSPECIFIED"]

    df["label"] = df["label"].apply(lambda x: [x])
    df = df.groupby("text").agg("sum").reset_index()
    df = df[["text", "label"]]

    df["label"] = df["label"].apply(lambda x: sorted(list(set(x))))

    return df


def _load_esco(task, validation_file, test_file):
    base_dir = "./data"
    esco_dir = os.path.join(base_dir, "esco")

    task_files = {
        "validation": os.path.join(esco_dir, validation_file)
        if validation_file
        else None,
        "test": os.path.join(esco_dir, test_file),
    }

    # get ontology
    # esco_skills, esco_descriptions, esco_priors = _load_esco_ontology(esco_dir)

    # get val and test set
    validation_df = (
        _prepare_esco_dataframe(task_files["validation"])
        if task_files["validation"]
        else None
    )
    test_df = _prepare_esco_dataframe(task_files["test"])

    return validation_df, test_df, None, None, None
