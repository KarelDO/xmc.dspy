import dspy

from src.utils import extract_labels_from_strings
from .config import IreraConfig
from .signatures import supported_signatures, supported_hints


class Infer(dspy.Module):
    def __init__(self, config: IreraConfig):
        super().__init__()
        self.config = config
        self.cot = dspy.ChainOfThoughtWithHint(
            supported_signatures[config.infer_signature_name]
        )
        self.cot.lm = config.infer_student_model
        self.hinter = supported_hints[config.infer_signature_name]

    def forward(self, text: str, label: list[str] = None) -> dspy.Prediction:
        parsed_outputs = set()

        hint = self.hinter(label) if label else None

        completion = self.cot(text=text, hint=hint).completions
        output = completion.output
        rationale = completion.rationale

        parsed_outputs.update(
            extract_labels_from_strings(output, do_lower=False, strip_punct=False)
        )

        parsed_outputs = list(parsed_outputs)

        return dspy.Prediction(predictions=parsed_outputs, rationale=rationale)
