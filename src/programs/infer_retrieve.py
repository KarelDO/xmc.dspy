import math
import json

import dspy
from .config import IreraConfig
from .retriever import Retriever
from .infer import Infer

import time

class InferRetrieve(dspy.Module):
    """Infer-Retrieve. Sets the Retriever, initializes the prior."""

    def __init__(
        self,
        config: IreraConfig,
    ):
        super().__init__()

        self.config = config

        # set LM predictor
        self.infer = Infer(config)

        # set retriever
        self.retriever = Retriever(config)

        # set prior and prior strength
        self.prior = self._set_prior(config.prior_path)
        self.prior_A = config.prior_A

    def forward(self, text: str, label: list[str] = None) -> dspy.Prediction:
        # Use the LM to predict label queries per chunk
        start = time.time()
        infer_output = self.infer(text, label=label)
        end = time.time()
        print(f"Infer time: {end - start}")
        preds = infer_output.predictions

        # Execute the queries against the label index and get the maximal score per label
        start = time.time()
        scores = self.retriever.retrieve(preds)
        end = time.time()
        print(f"Retrieve time: {end - start}")

        # Reweigh scores with prior statistics
        scores = self._update_scores_with_prior(scores)

        # Return the labels sorted
        labels = sorted(scores, key=lambda k: scores[k], reverse=True)

        return dspy.Prediction(
            predictions=labels, queries=preds, query_rationale=infer_output.rationale
        )

    def _set_prior(self, prior_path):
        """Loads the priors given a path and makes sure every term has a prior value (default value is 0)."""
        prior = json.load(open(prior_path, "r"))
        # Add 0 for every ontology term not in the file
        terms = self.retriever.ontology_terms
        terms_not_in_prior = set(terms).difference(set(prior.keys()))
        return prior | {t: 0.0 for t in terms_not_in_prior}

    def _update_scores_with_prior(self, scores: dict[str, float]) -> dict[str, float]:
        scores = {
            label: score * math.log(self.prior_A * self.prior[label] + math.e)
            for label, score in scores.items()
        }
        return scores
