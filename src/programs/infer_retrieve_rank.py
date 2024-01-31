import dspy
import json

from .infer_retrieve import InferRetrieve
from .config import IreraConfig
from .rank import Rank
from .chunking import Chunker


class InferRetrieveRank(dspy.Module):
    """Infer-Retrieve-Rank, as defined in https://arxiv.org/abs/2401.12178."""

    def __init__(
        self,
        config: IreraConfig,
    ):
        super().__init__()

        self.config = config

        # Set Chunker
        self.chunker = Chunker(config)

        # Set InferRetrieve
        self.infer_retrieve = InferRetrieve(config)

        # Set Rank
        self.rank = Rank(config)

        # Ranking hyperparameter
        self.rank_skip = config.rank_skip
        self.rank_topk = config.rank_topk

    def forward(self, text: str) -> dspy.Prediction:
        # NOTE: debugging prior
        scores = self.infer_retrieve.prior
        labels = sorted(scores, key=lambda k: scores[k], reverse=True)

        # # Take the first chunk
        # _, text = next(self.chunker(text))

        # # Get ranking from InferRetrieve
        # prediction = self.infer_retrieve(text)
        # labels = prediction.predictions

        # Get candidates
        options = labels[: self.rank_topk]

        # Rerank
        if not self.rank_skip:
            predictions = self.rank(text, options).predictions

            # Only keep options options that are valid
            selected_options = [o for o in predictions if o in options]

            print(f"Rank returned {len(selected_options)} valid options.")

            # Supplement options
            selected_options = selected_options + [
                o for o in options if o not in selected_options
            ]
        else:
            selected_options = options

        return dspy.Prediction(
            predictions=selected_options,
        )

    def dump_state(self):
        """Dump the state. Uses the DSPy dump_state but also adds the config file."""
        return super().dump_state() | {"config": self.config.to_dict()}

    def load_state(self, state: dict):
        super().load_state(state)

    @classmethod
    def from_state(cls, state: dict):
        # get the config
        config = IreraConfig.from_dict(state["config"])
        # create a new program
        program = cls(config)
        # load the state
        program.load_state(state)
        return program

    @classmethod
    def load(cls, path: str):
        state = json.load(open(path, "r"))
        return cls.from_state(state)

    def save(self, path: str):
        state = self.dump_state()
        with open(path, "w") as fp:
            json.dump(state, fp)
