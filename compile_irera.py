import os
import json

os.environ["DSP_NOTEBOOK_CACHEDIR"] = os.path.join(".", "local_cache")

from dspy import Models

from src.data_loaders import load_data
from src.programs import IreraConfig, InferRetrieveRank
from src.optimizers import OptimizerConfig, get_optimizer
from src.experiment import Experiment

from run_irera import run_irera

import argparse


def compile_irera(
    dataset_name: str,
    do_validation: bool,
    do_test: bool,
    irera_config: IreraConfig,
    optimizer_config: OptimizerConfig,
):
    # load data (all of these files needed for the config could be dumped separately in one folder)
    (
        train_examples,
        train_examples_with_label,
        validation_examples,
        test_examples,
        _,
        _,
        _,
    ) = load_data(dataset_name)

    # create program
    program = InferRetrieveRank(irera_config)

    # create optimizer
    optimizer = get_optimizer(optimizer_config)

    # Optimize
    program = optimizer.optimize(
        program, train_examples, train_examples_with_label, validation_examples
    )

    # Evaluate
    val_metrics, test_metrics, val_predictions, test_predictions = run_irera(
        program, validation_examples, test_examples, do_validation, do_test
    )

    exp = Experiment(
        dataset_name=dataset_name,
        program_name="infer-retrieve-rank",
        program_state=program.dump_state(),
        optimizer_config=optimizer_config.to_dict(),
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )
    return program, exp, val_predictions, test_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compile and evaluate Infer-Retrieve-Rank on an extreme multi-label classification (XMC) dataset."
    )

    # Add arguments
    parser.add_argument(
        "--lm_config_path",
        type=str,
        help="Specify the json containing the LM configs",
    )
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
    IreraConfig.add_arguments(parser)
    OptimizerConfig.add_argument(parser)

    # Parse the command-line arguments
    args = parser.parse_args()

    lm_config_path = args.lm_config_path
    dataset_name = args.dataset_name
    do_validation = args.do_validation
    do_test = args.do_test
    save_predictions = args.save_predictions

    Models(config_path=lm_config_path)

    # create configs
    irera_config = IreraConfig(**vars(args))
    optimizer_config = OptimizerConfig(**vars(args))

    print(f"lm_config_path: ", lm_config_path)
    print(f"dataset_name: ", dataset_name)
    print(f"do_validation: ", do_validation)
    print(f"do_test: ", do_test)
    print(f"save_predictions: ", save_predictions)
    print(f"irera_config: ", json.dumps(irera_config.to_dict(), indent=2))
    print(f"optimizer_config: ", json.dumps(optimizer_config.to_dict(), indent=2))

    program, exp, val_predictions, test_predictions = compile_irera(
        dataset_name,
        do_validation,
        do_test,
        irera_config,
        optimizer_config,
    )

    exp_path = exp.save("./results")
    if save_predictions:
        if val_predictions:
            path = os.path.join(*os.path.split(exp_path)[:-1], "val_predictions.json")
            with open(path, "w") as fp:
                json.dump(val_predictions, fp, indent=4)
        if test_predictions:
            path = os.path.join(*os.path.split(exp_path)[:-1], "test_predictions.json")
            with open(path, "w") as fp:
                json.dump(test_predictions, fp, indent=4)
