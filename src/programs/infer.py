import dspy

from src.utils import extract_labels_from_strings
from .config import IreraConfig
from .signatures import supported_signatures


class Infer(dspy.Module):
    def __init__(self, config: IreraConfig):
        super().__init__()
        self.config = config
        self.cot = dspy.ChainOfThought(
            supported_signatures[config.infer_signature_name]
        )

    def forward(self, text: str) -> dspy.Prediction:
        parsed_outputs = set()

        # NOTE: to improve downstream retrieval efficiency, we may want to normalize for slight variations in spelling / capitalization
        output = self.cot(
            text=text, config={"n": self.config.infer_num_predictions}
        ).completions.output
        parsed_outputs.update(
            extract_labels_from_strings(output, do_lower=False, strip_punct=False)
        )

        return dspy.Prediction(predictions=parsed_outputs)
