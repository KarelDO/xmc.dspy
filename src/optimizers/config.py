import json
from dspy import Models
from src.evaluators import supported_metrics


class Config:
    def __repr__(self):
        return self.to_dict().__repr__()

    def to_dict(self):
        """Convert the configuration to a dictionary."""
        dct = {
            key: value
            for key, value in vars(self).items()
            if not key.startswith("_") and not callable(value)
        }
        for k in dct:
            if isinstance(dct[k], Config):
                dct[k] = dct[k].to_dict()
        return dct

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


class CompilerConfig(Config):
    """Every option in config should be serializable. No attribute should start with '_', since these are not saved."""

    def __init__(self, **kwargs):
        # compilation names
        self.compiler_name = kwargs.pop("name")

        # compilation metrics
        self.metric_name = kwargs.pop("metric_name")

        # compilation kwargs
        self.max_bootstrapped_demos = kwargs.pop("max_bootstrapped_demos")
        self.max_labeled_demos = kwargs.pop("max_labeled_demos")
        self.num_candidate_programs = kwargs.pop("num_candidate_programs")

    @property
    def metric(self):
        return supported_metrics[self.metric_name]


class OptimizerConfig(Config):
    """Every option in config should be serializable. No attribute should start with '_', since these are not saved."""

    def __init__(self, **kwargs):
        # teacher models
        self.infer_teacher_model_name = kwargs.pop("infer_teacher_model_name")
        self.rank_teacher_model_name = kwargs.pop("rank_teacher_model_name")

        # compilation flags
        self.infer_compile = not kwargs.pop("no_infer_compile")
        self.infer_hint = kwargs.pop("infer_hint")
        self.rank_compile = not kwargs.pop("no_rank_compile")

        # compilation configs
        self.infer_compile_config = CompilerConfig(
            **{
                k.replace("infer_compile_", ""): v
                for k, v in kwargs.items()
                if "infer_compile_" in k
            }
        )
        self.rank_compile_config = CompilerConfig(
            **{
                k.replace("rank_compile_", ""): v
                for k, v in kwargs.items()
                if "rank_compile_" in k
            }
        )

        # optimizer
        self.optimizer_name = kwargs.pop("optimizer_name")

    @classmethod
    def add_argument(cls, parser):
        parser.add_argument("--infer_teacher_model_name", type=str)
        parser.add_argument("--rank_teacher_model_name", type=str)
        parser.add_argument(
            "--no_infer_compile",
            action="store_true",
            help="Specify if the Infer module should not be compiled (default: False)",
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
            "--infer_hint",
            action="store_true",
            help="If true, give the teacher model the true label in a hint and bootstrap a rational for the student (default: False)",
        )

        parser.add_argument(
            "--infer_compile_name",
            type=str,
            help="Name of the DSPy compiler for Infer.",
        )
        parser.add_argument("--infer_compile_max_bootstrapped_demos", type=int)
        parser.add_argument("--infer_compile_max_labeled_demos", default=0, type=int)
        parser.add_argument(
            "--infer_compile_num_candidate_programs", default=10, type=int
        )
        parser.add_argument(
            "--rank_compile_name", type=str, help="Name of the DSPy compiler for Rank."
        )
        parser.add_argument("--rank_compile_max_bootstrapped_demos", type=int)
        parser.add_argument("--rank_compile_max_labeled_demos", default=0, type=int)
        parser.add_argument(
            "--rank_compile_num_candidate_programs", default=10, type=int
        )
        parser.add_argument("--optimizer_name", type=str)

    @property
    def infer_teacher_model(self):
        return Models.get_lm(self.infer_teacher_model_name)

    @property
    def rank_teacher_model(self):
        return Models.get_lm(self.rank_teacher_model_name)
