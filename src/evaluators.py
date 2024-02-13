from dspy import Example
from dspy.evaluate import Evaluate
from src.metrics import *
import os

num_threads = int(os.environ.get("DSP_NUM_THREADS", 1))

""" Wrap metrics in an interface to be used by DSPy and create evaluators to easily run these metrics across a set of examples.
"""


# wrap the metrics for use in DSPy
def dspy_metric_rp50(gold: Example, pred, trace=None) -> float:
    return rp_at_k(gold.label, pred.predictions, 50)


def dspy_metric_rp10(gold: Example, pred, trace=None) -> float:
    return rp_at_k(gold.label, pred.predictions, 10)


def dspy_metric_rp5(gold: Example, pred, trace=None) -> float:
    return rp_at_k(gold.label, pred.predictions, 5)


def dspy_metric_rp1(gold: Example, pred, trace=None) -> float:
    return rp_at_k(gold.label, pred.predictions, 1)


def dspy_metric_recall10(gold: Example, pred, trace=None) -> float:
    return recall_at_k(gold.label, pred.predictions, 10)


def dspy_metric_recall5(gold: Example, pred, trace=None) -> float:
    return recall_at_k(gold.label, pred.predictions, 5)


def dspy_metric_recall1(gold: Example, pred, trace=None) -> float:
    return recall_at_k(gold.label, pred.predictions, 1)


# Experimental metric for Rank module
def rank_gain_recall10(gold: Example, pred, trace=None) -> float:
    # Check the RP10 of the IRe module
    recall10_ire = recall_at_k(gold.label, pred.retrieve_outputs, k=10)
    recall10_irera = recall_at_k(gold.label, pred.predictions, k=10)

    rank_gain = recall10_irera - recall10_ire
    rank_success = rank_gain > 0
    return rank_success


def create_evaluators(examples):
    # create a suite of DSPy evaluators based on a set of examples
    evaluate_recall10 = Evaluate(
        devset=examples,
        metric=dspy_metric_recall10,
        num_threads=num_threads,
        display_progress=False,
        display_table=0,
        max_errors=100,
    )
    evaluate_rp10 = Evaluate(
        devset=examples,
        metric=dspy_metric_rp10,
        num_threads=num_threads,
        display_progress=False,
        display_table=0,
        max_errors=100,
    )
    evaluate_recall5 = Evaluate(
        devset=examples,
        metric=dspy_metric_recall5,
        num_threads=num_threads,
        display_progress=False,
        display_table=0,
        max_errors=100,
    )
    evaluate_rp5 = Evaluate(
        devset=examples,
        metric=dspy_metric_rp5,
        num_threads=num_threads,
        display_progress=False,
        display_table=0,
        max_errors=100,
    )
    evaluate_rp50 = Evaluate(
        devset=examples,
        metric=dspy_metric_rp50,
        num_threads=num_threads,
        display_progress=False,
        display_table=0,
        max_errors=100,
    )
    return {
        "recall10": evaluate_recall10,
        "recall5": evaluate_recall5,
        "rp50": evaluate_rp50,
        "rp10": evaluate_rp10,
        "rp5": evaluate_rp5,
    }


supported_metrics = {
    "rp5": dspy_metric_rp5,
    "rp10": dspy_metric_rp10,
    "rp50": dspy_metric_rp50,
    "recall5": dspy_metric_recall5,
    "recall10": dspy_metric_recall10,
    "rank_gain_recall10": rank_gain_recall10,
}
