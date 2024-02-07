import pandas as pd
import dspy
import random
import argparse

from typing import Union

from .biodex import _load_biodex
from .esco import _load_esco


def get_dspy_examples(
    validation_df: Union[pd.DataFrame, None],
    test_df: pd.DataFrame,
    n_validation: int = None,
    n_test: int = None,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    valset, testset = [], []

    n_validation = float("+inf") if not n_validation else n_validation
    n_test = float("+inf") if not n_test else n_test

    if validation_df is not None:
        for _, example in validation_df.iterrows():
            if len(valset) >= n_validation:
                break
            valset.append(example.to_dict())
        valset = [dspy.Example(**x).with_inputs("text") for x in valset]
        # valsetX = [dspy.Example(**x).with_inputs('text', 'label') for x in valset]

    for _, example in test_df.iterrows():
        if len(testset) >= n_test:
            break
        testset.append(example.to_dict())
    testset = [dspy.Example(**x).with_inputs("text") for x in testset]
    # testsetX = [dspy.Example(**x).with_inputs('text', 'label') for x in testset]

    # print(len(valset), len(testset))
    return valset, testset


def load_data(dataset: str):
    if dataset == "esco_house":
        (
            validation_df,
            test_df,
            ontology_items,
            ontology_descriptions,
            ontology_prior,
        ) = _load_esco(
            "house", "house_validation_annotations.csv", "house_test_annotations.csv"
        )
    elif dataset == "esco_tech":
        (
            validation_df,
            test_df,
            ontology_items,
            ontology_descriptions,
            ontology_prior,
        ) = _load_esco(
            "tech", "tech_validation_annotations.csv", "tech_test_annotations.csv"
        )
    elif dataset == "esco_techwolf":
        (
            validation_df,
            test_df,
            ontology_items,
            ontology_descriptions,
            ontology_prior,
        ) = _load_esco("techwolf", None, "techwolf_test_annotations.csv")
    elif dataset == "biodex_reactions":
        (
            validation_df,
            test_df,
            ontology_items,
            ontology_descriptions,
            ontology_prior,
        ) = _load_biodex()
    else:
        raise ValueError("Dataset not supported.")

    validation_examples, test_examples = get_dspy_examples(validation_df, test_df)

    # shuffle
    # NOTE: pull out this seed to get confidence intervals
    random.seed(42)
    random.shuffle(validation_examples)
    random.shuffle(test_examples)

    # log some stats
    print(f"Dataset: {dataset}")

    print(f"# {dataset}: Total Validation size: {len(validation_examples)}")
    print(f"# {dataset}: Total Test size: {len(test_examples)}")
    # print(f"# {dataset}: Ontology items: {len(ontology_items)}")
    if "techwolf" not in dataset:
        print(
            f'{dataset}: avg # ontology items per input (for validation set): {round(validation_df["label"].apply(len).mean(),2)}'
        )
        print(
            f'{dataset}: Q25, Q50, Q75, Q95 # ontology items per input (for validation set): {validation_df["label"].apply(len).quantile([0.25, 0.5, 0.75, 0.95])}'
        )
    else:
        print(
            f'{dataset}: avg # ontology items per input (for test set): {round(test_df["label"].apply(len).mean(),2)}'
        )
        print(
            f'{dataset}: Q25, Q50, Q75, Q95 # ontology items per input (for test set): {test_df["label"].apply(len).quantile([0.25, 0.5, 0.75, 0.95])}'
        )

    # split off some of the validation examples for demonstrations
    # TODO: put these rangers in a config somewhere, or automate with random seed.
    if dataset == "esco_house" or dataset == "esco_tech":
        train_examples = validation_examples[:10]
        validation_examples = validation_examples[10:]
    elif dataset == "esco_techwolf":
        # techwolf has no validation data, use a mix of house and tech as proxy
        house_train, house_val, _, _, _, _ = load_data("esco_house")
        # tech_train, tech_val, _, _, _, _ = load_data("esco_tech")
        # train_examples = house_train + tech_train
        train_examples = house_train
        # validation_examples = house_val + tech_val
        validation_examples = house_val
        # shuffle train and val again
        random.seed(42)
        random.shuffle(train_examples)
        random.shuffle(validation_examples)
    elif dataset == "biodex_reactions":
        # train_examples = validation_examples[:100]
        # validation_examples = validation_examples[100:200]
        # test_examples = test_examples[:500]
        train_examples = validation_examples[:10]
        validation_examples = validation_examples[100:150]
        test_examples = test_examples[:250]

    print(f"{dataset}: # Used Train size: {len(train_examples)}")
    print(f"{dataset}: # Used Validation size: {len(validation_examples)}")
    print(f"{dataset}: # Used Test size: {len(test_examples)}")

    # Create train_examples with label inputs
    train_examples_label = [x.with_inputs("text", "label") for x in train_examples]

    return (
        train_examples,
        train_examples_label,
        validation_examples,
        test_examples,
        ontology_items,
        ontology_descriptions,
        ontology_prior,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataloader")

    # Add arguments
    parser.add_argument(
        "--dataset",
        type=str,
        help="Specify the dataset",
    )
    args = parser.parse_args()

    (
        train_examples,
        validation_examples,
        test_examples,
        ontology_items,
        ontology_descriptions,
        ontology_prior,
    ) = load_data(args.dataset)
    print("Done.")
