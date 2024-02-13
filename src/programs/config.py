import json
from dspy import Models


class IreraConfig:
    """Every option in config should be serializable. No attribute should start with '_', since these are not saved."""

    def __init__(self, **kwargs):
        # signatures
        self.infer_signature_name = kwargs.pop("infer_signature_name")
        self.rank_signature_name = kwargs.pop("rank_signature_name")

        # lms
        self.infer_student_model_name = kwargs.pop("infer_student_model_name")
        self.rank_student_model_name = kwargs.pop("rank_student_model_name")

        # hyperparameters
        self.prior_A = kwargs.pop("prior_A", 0)
        self.prior_path = kwargs.pop("prior_path", None)
        self.rank_topk = kwargs.pop("rank_topk", 50)
        self.chunk_context_window = kwargs.pop("chunk_context_window", 3000)
        self.chunk_max_windows = kwargs.pop("chunk_max_windows", 5)
        self.chunk_window_overlap = kwargs.pop("chunk_window_overlap", 0.02)

        # program logic flow
        self.rank_skip = kwargs.pop("no_rank", False)

        # ontology
        self.ontology_path = kwargs.pop("ontology_path", None)
        self.ontology_name = kwargs.pop("ontology_name", None)
        self.retriever_model_name = kwargs.pop(
            "retriever_model_name", "sentence-transformers/all-mpnet-base-v2"
        )

        # optimizer
        self.optimizer_name = kwargs.pop("optimizer_name", None)

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            "--retriever_model_name",
            type=str,
            default="sentence-transformers/all-mpnet-base-v2",
            help="Specify the retriever model name (default: sentence-transformers/all-mpnet-base-v2)",
        )
        parser.add_argument("--infer_signature_name", type=str)
        parser.add_argument("--rank_signature_name", type=str)
        parser.add_argument("--infer_student_model_name", type=str)
        parser.add_argument("--rank_student_model_name", type=str)
        parser.add_argument(
            "--no_rank",
            action="store_true",
            help="Specify if the Rank module should be ablated (default: False)",
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
            "--prior_path",
            type=str,
            help="Path to the JSON file containing prior data.",
        )
        parser.add_argument(
            "--ontology_path", type=str, help="Path to the ontology file."
        )
        parser.add_argument("--ontology_name", type=str, help="Name of the ontology.")
        parser.add_argument("--chunk_context_window", type=int, default=3000)

    @property
    def infer_student_model(self):
        return Models.get_lm(self.infer_student_model_name)

    @property
    def rank_student_model(self):
        return Models.get_lm(self.rank_student_model_name)

    def __repr__(self):
        return self.to_dict().__repr__()

    def to_dict(self):
        """Convert the configuration to a dictionary."""
        return {
            key: value
            for key, value in vars(self).items()
            if not key.startswith("_") and not callable(value)
        }

    def to_json(self, filename):
        """Save the configuration to a JSON file."""
        with open(filename, "w") as file:
            json.dump(self.to_dict(), file, indent=4)

    @classmethod
    def from_dict(cls, config_dict):
        """Create an instance of the configuration from a dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_json(cls, filename):
        """Load the configuration from a JSON file."""
        with open(filename, "r") as file:
            config_dict = json.load(file)
        return cls.from_dict(config_dict)
