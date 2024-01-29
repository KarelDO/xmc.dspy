import dspy
from src.utils import extract_labels_from_strings
from .config import IreraConfig
from .signatures import supported_signatures


class Rank(dspy.Module):
    def __init__(self, config: IreraConfig):
        super().__init__()

        self.config = config
        self.cot = dspy.ChainOfThought(supported_signatures[config.rank_signature_name])

    def forward(self, text: str, options: list[str]) -> dspy.Predict:
        parsed_outputs = []

        output = self.cot(text=text, options=options).completions.output

        parsed_outputs = extract_labels_from_strings(
            output, do_lower=False, strip_punct=False, split_colon=True
        )

        return dspy.Prediction(predictions=parsed_outputs)
