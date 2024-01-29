from dataclasses import dataclass, asdict
from os import path, mkdir
from json import dump


@dataclass
class Experiment:
    """Keep track of experiment configuration, optimization parameters, and the resulting program state. Handles saving results."""

    # experiment configuration
    dataset_name: str
    program_name: str
    infer_student_model_name: str
    infer_teacher_model_name: str
    rank_student_model_name: str
    rank_teacher_model_name: str
    infer_compile: bool
    infer_compile_metric_name: str
    rank_compile: bool
    rank_compile_metric_name: str
    optimizer_name: str

    # results
    validation_rp5: float = None
    validation_recall5: float = None
    validation_rp10: float = None
    validation_rp50: float = None
    validation_recall10: float = None
    test_rp5: float = None
    test_recall5: float = None
    test_rp10: float = None
    test_rp50: float = None
    test_recall10: float = None

    # program state
    program_state: dict = None

    def get_name(self, index: int):
        name = f"{self.dataset_name}_{self.program_name}_{index:02d}"
        return name

    def save(self, results_dir: str):
        index = 0
        if not path.exists(results_dir):
            mkdir(results_dir)
        name = self.get_name(index)
        file = path.join(results_dir, name)
        # iterate the index if the experiment already exists
        while path.exists(file):
            index += 1
            name = self.get_name(index)
            file = path.join(results_dir, name)

        # make dir
        mkdir(file)

        # save program_state separately
        to_save = asdict(self)
        program_state = to_save.pop("program_state")

        # save results
        results_file = path.join(file, "results.json")
        with open(results_file, "w") as fp:
            dump(to_save, fp, indent=4)

        # save state
        state_file = path.join(file, "program_state.json")
        with open(state_file, "w") as fp:
            dump(program_state, fp, indent=4)
