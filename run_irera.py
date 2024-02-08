import os
import json

os.environ["DSP_NOTEBOOK_CACHEDIR"] = os.path.join(".", "local_cache")

from dspy import Models
from src.data_loaders import load_data
from src.evaluators import create_evaluators
from src.programs import InferRetrieveRank

import argparse


def run_irera(program, validation_examples, test_examples, do_validation, do_test):
    val_metrics, test_metrics, val_predictions, test_predictions = (
        None,
        None,
        None,
        None,
    )

    # Validate / Test
    if do_validation:
        print("validating program...")
        validation_evaluators = create_evaluators(validation_examples)
        validation_rp50, val_predictions = validation_evaluators["rp50"](
            program, return_predictions=True
        )
        validation_rp10, scores_val_rp10 = validation_evaluators["rp10"](
            program, return_all_scores=True
        )
        validation_rp5, scores_val_rp5 = validation_evaluators["rp5"](
            program, return_all_scores=True
        )

        for dct, rp10, rp5 in zip(val_predictions, scores_val_rp10, scores_val_rp5):
            dct["correct_rp10"] = rp10
            dct["correct_rp5"] = rp5

        val_metrics = {
            "rp50": validation_rp50,
            "rp10": validation_rp10,
            "rp5": validation_rp5,
        }

    if do_test:
        print("testing program...")
        test_evaluators = create_evaluators(test_examples)
        test_rp50, test_predictions = test_evaluators["rp50"](
            program, return_predictions=True
        )
        test_rp10, scores_test_rp10 = test_evaluators["rp10"](
            program, return_all_scores=True
        )
        test_rp5, scores_test_rp5 = test_evaluators["rp5"](
            program, return_all_scores=True
        )

        for dct, rp10, rp5 in zip(test_predictions, scores_test_rp10, scores_test_rp5):
            dct["correct_rp10"] = rp10
            dct["correct_rp5"] = rp5

        test_metrics = {
            "rp50": test_rp50,
            "rp10": test_rp10,
            "rp5": test_rp5,
        }

    if do_validation:
        print("Final program validation_rp50: ", validation_rp50)
        print("Final program validation_rp10: ", validation_rp10)
        print("Final program validation_rp5: ", validation_rp5)

    if do_test:
        print("Final program test_rp50: ", test_rp50)
        print("Final program test_rp10: ", test_rp10)
        print("Final program test_rp5: ", test_rp5)

    return val_metrics, test_metrics, val_predictions, test_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Infer-Retrieve-Rank on an extreme multi-label classification (XMC) dataset."
    )

    # Add arguments
    parser.add_argument("--state_path", type=str)
    parser.add_argument("--lm_config_path", type=str)
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Specify the dataset",
    )
    parser.add_argument(
        "--do_validation",
        action="store_true",
        help="Specify if validation results need to be calculated (default: False)",
    )
    parser.add_argument(
        "--do_test",
        action="store_true",
        help="Specify if test results need to be calculated (default: False)",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Specify if predictions need to be saved (default: False)",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    state_path = args.state_path
    lm_config_path = args.lm_config_path
    dataset_name = args.dataset_name
    do_validation = args.do_validation
    do_test = args.do_test
    save_predictions = args.save_predictions

    Models(config_path=lm_config_path)

    print("state_path: ", state_path)
    print("lm_config_path: ", lm_config_path)
    print("dataset_name: ", dataset_name)
    print("do_validation: ", do_validation)
    print("do_test: ", do_test)
    print(f"save_predictions: ", save_predictions)

    # load data
    (
        _,
        _,
        validation_examples,
        test_examples,
        _,
        _,
        _,
    ) = load_data(dataset_name)

    # load program
    program = InferRetrieveRank.load(state_path)

    # evaluate
    val_metrics, test_metrics, val_predictions, test_predictions = run_irera(
        program, validation_examples, test_examples, do_validation, do_test
    )

    if save_predictions:
        if val_predictions:
            path = os.path.join(*os.path.split(state_path)[:-1], "val_predictions.json")
            with open(path, "w") as fp:
                json.dump(val_predictions, fp, indent=4)
        if test_predictions:
            path = os.path.join(
                *os.path.split(state_path)[:-1], "test_predictions.json"
            )
            with open(path, "w") as fp:
                json.dump(test_predictions, fp, indent=4)
