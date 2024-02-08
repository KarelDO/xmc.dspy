import dspy
from collections import defaultdict

""" To add signatures or hints, subclass `dspy.Signature` and add the signature to `supported_signatures` with a short-hand for easy access throughout the code.
"""


def InferHintESCO(output):
    # hint = f"Produce the rationale to find the correct skills. The correct skills are: {', '.join(output)}."
    # hint = f"Produce a simple rationale to find the correct skills. The correct skills are: {', '.join(output)}."
    # hint = f"Produce a simple rationale to find the correct skills. The correct skills should include: {', '.join(output)}."
    hint = f"Produce the reasoning to find the correct skills. Return all applicable skill queries. The correct skills should include: {', '.join(output)}. Add any other skills you think are relevant."
    return hint


class InferSignatureESCO(dspy.Signature):
    __doc__ = f"""Given a snippet from a job vacancy, identify all the ESCO job skills mentioned. Always return skills."""

    text = dspy.InputField(prefix="Vacancy:")
    output = dspy.OutputField(
        prefix="Skills:",
        desc="list of comma-separated ESCO skills",
        format=lambda x: ", ".join(x) if isinstance(x, list) else x,
    )


class RankSignatureESCO(dspy.Signature):
    __doc__ = f"""Given a snippet from a job vacancy, pick the 10 most applicable skills from the options that are directly expressed in the snippet."""

    text = dspy.InputField(prefix="Vacancy:")
    options = dspy.InputField(
        prefix="Options:",
        desc="List of comma-separated options to choose from",
        format=lambda x: ", ".join(x) if isinstance(x, list) else x,
    )
    output = dspy.OutputField(
        prefix="Skills:",
        desc="list of comma-separated ESCO skills",
        format=lambda x: ", ".join(x) if isinstance(x, list) else x,
    )


class InferSignatureBioDEX(dspy.Signature):
    __doc__ = f"""Given a snippet from a medical article, identify the adverse drug reactions affecting the patient. Always return reactions."""

    text = dspy.InputField(prefix="Article:")
    output = dspy.OutputField(
        prefix="Reactions:",
        desc="list of comma-separated adverse drug reactions",
        format=lambda x: ", ".join(x) if isinstance(x, list) else x,
    )


class RankSignatureBioDEX(dspy.Signature):
    __doc__ = f"""Given a snippet from a medical article, pick the 10 most applicable adverse reactions from the options that are directly expressed in the snippet."""

    text = dspy.InputField(prefix="Article:")
    options = dspy.InputField(
        prefix="Options:",
        desc="List of comma-separated options to choose from",
        format=lambda x: ", ".join(x) if isinstance(x, list) else x,
    )
    output = dspy.OutputField(
        prefix="Reactions:",
        desc="list of comma-separated adverse drug reactions",
        format=lambda x: ", ".join(x) if isinstance(x, list) else x,
    )


supported_signatures = {
    "infer_esco": InferSignatureESCO,
    "rank_esco": RankSignatureESCO,
    "infer_biodex": InferSignatureBioDEX,
    "rank_biodex": RankSignatureBioDEX,
}

supported_hints = defaultdict(lambda x: None, {"infer_esco": InferHintESCO})
