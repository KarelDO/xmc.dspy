import os

os.environ["DSP_NOTEBOOK_CACHEDIR"] = os.path.join(".", "local_cache")

from dspy import Models

from src.data_loaders import load_data
from src.programs import IreraConfig, InferRetrieveRank
from src.optimizer import supported_optimizers
from src.experiment import Experiment
from src.evaluators import create_evaluators

import argparse


def compile_irera(
    dataset_name: str,
    retriever_model_name: str,
    infer_signature_name: str,
    infer_student_model_name: str,
    infer_teacher_model_name: str,
    rank_signature_name: str,
    rank_student_model_name: str,
    rank_teacher_model_name: str,
    infer_compile: bool,
    infer_compile_metric_name: str,
    rank_skip: bool,
    rank_compile: bool,
    rank_compile_metric_name: str,
    prior_A: int,
    rank_topk: int,
    do_validation: bool,
    do_test: bool,
    prior_path: str,
    ontology_path: str,
    ontology_name: str,
    optimizer_name: str,
):
    # Create config
    config = IreraConfig(
        infer_signature_name=infer_signature_name,
        rank_signature_name=rank_signature_name,
        prior_A=prior_A,
        prior_path=prior_path,
        rank_topk=rank_topk,
        rank_skip=rank_skip,
        ontology_path=ontology_path,
        ontology_name=ontology_name,
        retriever_model_name=retriever_model_name,
        optimizer_name=optimizer_name,
    )

    # load data (all of these files needed for the config could be dumped separately in one folder)
    (
        train_examples,
        validation_examples,
        test_examples,
        _,
        _,
        _,
    ) = load_data(dataset_name)

    # create program
    program = InferRetrieveRank(config)

    # set program students
    modules_to_lms = {
        "infer_retrieve.infer": {
            "teacher": Models.get_lm(infer_teacher_model_name),
            "student": Models.get_lm(infer_student_model_name),
        },
        "rank": {
            "teacher": Models.get_lm(rank_teacher_model_name),
            "student": Models.get_lm(rank_student_model_name),
        },
    }

    program.infer_retrieve.infer.cot.lm = modules_to_lms["infer_retrieve.infer"][
        "student"
    ]
    program.rank.cot.lm = modules_to_lms["rank"]["student"]

    # create optimizer
    optimizer_class = supported_optimizers[config.optimizer_name]
    optimizer_kwargs = {
        "modules_to_lms": modules_to_lms,
        "infer_compile": infer_compile,
        "infer_compile_metric_name": infer_compile_metric_name,
        "rank_compile": rank_compile,
        "rank_compile_metric_name": rank_compile_metric_name,
    }
    optimizer = optimizer_class(**optimizer_kwargs)

    # Optimize
    program = optimizer.optimize(
        program, train_examples, validation_examples=validation_examples
    )

    # Validate / Test
    if do_validation:
        print("validating final program...")
        validation_evaluators = create_evaluators(validation_examples)
        validation_rp50 = validation_evaluators["rp50"](program)
        validation_rp10 = validation_evaluators["rp10"](program)
        validation_rp5 = validation_evaluators["rp5"](program)

    if do_test:
        print("testing final program...")
        test_evaluators = create_evaluators(test_examples)
        test_rp10 = test_evaluators["rp10"](program)
        test_rp5 = test_evaluators["rp5"](program)

    if do_validation:
        print("Final program validation_rp50: ", validation_rp50)
        print("Final program validation_rp10: ", validation_rp10)
        print("Final program validation_rp5: ", validation_rp5)

    if do_test:
        print("Final program test_rp10: ", test_rp10)
        print("Final program test_rp5: ", test_rp5)

    exp = Experiment(
        dataset_name=dataset_name,
        program_name="infer-retrieve-rank",
        infer_student_model_name=infer_student_model_name,
        infer_teacher_model_name=infer_teacher_model_name,
        rank_student_model_name=rank_student_model_name,
        rank_teacher_model_name=rank_teacher_model_name,
        infer_compile=infer_compile,
        infer_compile_metric_name=infer_compile_metric_name,
        rank_compile=rank_compile,
        rank_compile_metric_name=rank_compile_metric_name,
        validation_rp5=validation_rp5 if do_validation else None,
        validation_rp10=validation_rp10 if do_validation else None,
        validation_rp50=validation_rp50 if do_validation else None,
        test_rp5=test_rp5 if do_test else None,
        test_rp10=test_rp10 if do_test else None,
        program_state=program.dump_state(),
        optimizer_name=optimizer_name,
    )
    return exp, program


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
        "--retriever_model_name",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Specify the retriever model name (default: sentence-transformers/all-mpnet-base-v2)",
    )
    parser.add_argument("--infer_signature_name", type=str)
    parser.add_argument("--rank_signature_name", type=str)
    parser.add_argument("--infer_student_model_name", type=str)
    parser.add_argument("--infer_teacher_model_name", type=str)
    parser.add_argument("--rank_student_model_name", type=str)
    parser.add_argument("--rank_teacher_model_name", type=str)
    parser.add_argument(
        "--no_infer_compile",
        action="store_true",
        help="Specify if the Infer module should not be compiled (default: False)",
    )
    parser.add_argument(
        "--no_rank",
        action="store_true",
        help="Specify if the Rank module should be ablated (default: False)",
    )
    parser.add_argument(
        "--no_rank_compile",
        action="store_true",
        help="Specify if the Rank module should not be compiled (default: False)",
    )
    parser.add_argument(
        "--infer_compile_metric_name",
        default="rp10",
        help="Specify for which metric the system should be compiled (default: rp10)",
    )
    parser.add_argument(
        "--rank_compile_metric_name",
        default="rp10",
        help="Specify for which metric the system should be compiled (default: rp10)",
    )
    parser.add_argument(
        "--prior_A",
        default=0,
        type=int,
        help="Specify influence of prior statistics on predicting reranking (default: 0)",
    )
    parser.add_argument(
        "--rank_topk",
        default=50,
        type=int,
        help="Specify how many the top k options that are input to the Rank module for reranking (default: 50)",
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
        "--prior_path", type=str, help="Path to the JSON file containing prior data."
    )

    parser.add_argument("--ontology_path", type=str, help="Path to the ontology file.")

    parser.add_argument("--ontology_name", type=str, help="Name of the ontology.")
    parser.add_argument("--optimizer_name", type=str, help="Name of the ontology.")

    # parser.add_argument(
    #     "--max_windows",
    #     default=1,
    #     type=int,
    #     help="Specify total amount of chunking windows to use (default: 1)",
    # )

    # Parse the command-line arguments
    args = parser.parse_args()

    # NOTE: use a config object.
    lm_config_path = args.lm_config_path
    dataset_name = args.dataset_name
    retriever_model_name = args.retriever_model_name
    infer_signature_name = args.infer_signature_name
    infer_student_model_name = args.infer_student_model_name
    infer_teacher_model_name = args.infer_teacher_model_name
    rank_signature_name = args.rank_signature_name
    rank_student_model_name = args.rank_student_model_name
    rank_teacher_model_name = args.rank_teacher_model_name
    infer_compile = not args.no_infer_compile
    infer_compile_metric_name = args.infer_compile_metric_name
    rank_skip = args.no_rank
    rank_compile = not args.no_rank_compile
    rank_compile_metric_name = args.rank_compile_metric_name
    prior_A = args.prior_A
    rank_topk = args.rank_topk
    do_validation = args.do_validation
    do_test = args.do_test
    prior_path = args.prior_path
    ontology_path = args.ontology_path
    ontology_name = args.ontology_name
    optimizer_name = args.optimizer_name

    print(f"dataset_name: ", dataset_name)
    print(f"retriever_model_name: ", retriever_model_name)
    print(f"infer_signature_name: ", infer_signature_name)
    print(f"infer_student_model_name: ", infer_student_model_name)
    print(f"infer_teacher_model_name: ", infer_teacher_model_name)
    print(f"rank_signature_name: ", rank_signature_name)
    print(f"rank_student_model_name: ", rank_student_model_name)
    print(f"rank_teacher_model_name: ", rank_teacher_model_name)
    print(f"infer_compile: ", infer_compile)
    print(f"infer_compile_metric_name: ", infer_compile_metric_name)
    print(f"rank_skip: ", rank_skip)
    print(f"rank_compile: ", rank_compile)
    print(f"rank_compile_metric_name: ", rank_compile_metric_name)
    print(f"prior_A: ", prior_A)
    print(f"rank_topk: ", rank_topk)
    print(f"do_validation: ", do_validation)
    print(f"do_test: ", do_test)
    print(f"prior_path: ", prior_path)
    print(f"ontology_path: ", ontology_path)
    print(f"ontology_name: ", ontology_name)
    print(f"optimizer_name: ", optimizer_name)

    Models(config_path=lm_config_path)

    experiment, program = compile_irera(
        dataset_name,
        retriever_model_name,
        infer_signature_name,
        infer_student_model_name,
        infer_teacher_model_name,
        rank_signature_name,
        rank_student_model_name,
        rank_teacher_model_name,
        infer_compile,
        infer_compile_metric_name,
        rank_skip,
        rank_compile,
        rank_compile_metric_name,
        prior_A,
        rank_topk,
        do_validation,
        do_test,
        prior_path,
        ontology_path,
        ontology_name,
        optimizer_name,
    )
    experiment.save("./results")
