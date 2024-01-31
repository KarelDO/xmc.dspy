import math
import json

import dspy
from .config import IreraConfig
from .retriever import Retriever
from .infer import Infer


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

    def forward(self, text: str) -> dspy.Prediction:
        # Use the LM to predict label queries per chunk
        preds = self.infer(text).predictions

        # Execute the queries against the label index and get the maximal score per label
        scores = self.retriever.retrieve(preds)

        # Reweigh scores with prior statistics
        scores = self._update_scores_with_prior(scores)

        # Return the labels sorted
        labels = sorted(scores, key=lambda k: scores[k], reverse=True)

        return dspy.Prediction(
            predictions=labels,
        )

    def _set_prior(self, prior_path):
        """Loads the priors given a path and makes sure every term has a prior value (default value is 0)."""
        prior = json.load(open(prior_path, "r"))
        # Add 0 for every ontology term not in the file
        terms = self.retriever.ontology_terms
        terms_not_in_prior = set(terms).difference(set(prior.keys()))
        return prior | {t: 0.0 for t in terms_not_in_prior}

    def _update_scores_with_prior(self, scores: dict[str, float]) -> dict[str, float]:
        updated_scores = {
            # label: score * math.log(self.prior_A * self.prior[label] + math.e)
            label: score + math.log(self.prior_A * self.prior[label] + math.e)
            for label, score in scores.items()
        }

        # NOTE: Debugging rank movement
        scores_sorted = sorted(scores, key=scores.get, reverse=True)
        updated_scores_sorted = sorted(
            updated_scores, key=updated_scores.get, reverse=True
        )
        print(
            "Average Rank Move: ",
            _average_rank_move(scores_sorted, updated_scores_sorted),
        )

        print(
            "Items that changed in top50: ",
            len(set(scores_sorted[:50]).difference(set(updated_scores_sorted[:50]))),
        )
        print(
            "Items that changed in top100: ",
            len(set(scores_sorted[:100]).difference(set(updated_scores_sorted[:100]))),
        )

        # NOTE: just return prior for now
        # return updated_scores
        return self.prior


# NOTE: Debugging rank movement
import numpy as np


def _average_rank_move(ordering1, ordering2):
    # Create a mapping of items to their ranks in each ordering
    ranks1 = {item: rank for rank, item in enumerate(ordering1)}
    ranks2 = {item: rank for rank, item in enumerate(ordering2)}

    # Get the ranks for each item in both orderings
    ranks_ordering1 = np.array([ranks1[item] for item in ordering1])
    ranks_ordering2 = np.array(
        [ranks2[item] for item in ordering1]
    )  # Use ordering1 for consistent item order

    # Calculate the absolute differences in ranks
    rank_moves = np.abs(ranks_ordering1 - ranks_ordering2)

    # Calculate the average rank move
    average_rank_move = np.mean(rank_moves)

    return average_rank_move
