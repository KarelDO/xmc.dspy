import dspy
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from src.programs import InferRetrieveRank
from src.evaluators import supported_metrics
from math import floor


# NOTE: not actually left-to-right because of DSPy bug.
# NOTE: resolve code duplication with inheriting classes
class LeftToRightOptimizer:
    def __init__(
        self,
        modules_to_lms: dict[str, tuple],
        infer_compile: bool,
        infer_compile_metric_name: str,
        rank_compile: bool,
        rank_compile_metric_name: str,
    ):
        # TODO: add an optimization config
        self.modules_to_lms = modules_to_lms

        self.infer_compile = infer_compile
        self.infer_compile_metric = supported_metrics[infer_compile_metric_name]

        self.rank_compile = rank_compile
        self.rank_compile_metric = supported_metrics[rank_compile_metric_name]

        # compilation hyperparameters
        self.max_bootstrapped_demos = 2
        self.max_labeled_demos = 0
        self.max_rounds = 1
        self.num_candidate_programs = 10
        self.num_threads = 8

    def create_compiler(self, metric):
        return BootstrapFewShotWithRandomSearch(
            metric=metric,
            max_bootstrapped_demos=self.max_bootstrapped_demos,
            max_labeled_demos=self.max_labeled_demos,
            max_rounds=self.max_rounds,
            num_candidate_programs=self.num_candidate_programs,
            num_threads=self.num_threads,
        )

    # NOTE: ideally, optimize should be agnostic of program implementation.
    # NOTE: it should get an optimization config.
    def optimize(
        self,
        program: InferRetrieveRank,
        train_examples: list[dspy.Example],
        validation_examples: list[dspy.Example],
    ) -> dspy.Module:
        # First round
        if self.infer_compile:
            # Create first-round teacher
            teacher = program.deepcopy()
            teacher.infer_retrieve.infer.cot.lm = self.modules_to_lms[
                "infer_retrieve.infer"
            ]["teacher"]

            # No ranking in first-round
            rank_skipped = program.rank_skip
            teacher.rank_skip = True
            program.rank_skip = True

            # create compiler
            infer_compiler = self.create_compiler(
                self.infer_compile_metric,
            )

            # compile
            program = infer_compiler.compile(
                program,
                teacher=teacher,
                trainset=train_examples,
                valset=validation_examples,
                restrict=range(20),
            )

            # Set back Rank module to what it originally was.
            program.rank_skip = rank_skipped

            # Freeze module we just compiled
            # NOTE: this may not be working actually.
            program.infer_retrieve._compiled = True
            program._compiled = False

        # Second round
        if self.rank_compile and not program.rank_skip:
            # Create second-round teacher
            teacher = program.deepcopy()
            teacher.rank.cot.lm = self.modules_to_lms["rank"]["teacher"]

            rank_compiler = self.create_compiler(self.rank_compile_metric)

            # compile
            program = rank_compiler.compile(
                program,
                teacher=teacher,
                trainset=train_examples,
                valset=validation_examples,
                restrict=range(20),
            )

        return program


class End2EndOptimizer:
    def __init__(
        self,
        modules_to_lms: dict[str, tuple],
        infer_compile: bool,
        infer_compile_metric_name: str,
        rank_compile: bool,
        rank_compile_metric_name: str,
    ):
        # TODO: add an optimization config
        self.modules_to_lms = modules_to_lms

        self.infer_compile = infer_compile
        self.infer_compile_metric = supported_metrics[infer_compile_metric_name]

        self.rank_compile = rank_compile
        self.rank_compile_metric = supported_metrics[rank_compile_metric_name]

        # compilation hyperparameters
        self.max_bootstrapped_demos = 2
        self.max_labeled_demos = 0
        self.max_rounds = 1
        self.num_candidate_programs = 10
        self.num_threads = 8

    def create_compiler(self, metric):
        return BootstrapFewShotWithRandomSearch(
            metric=metric,
            max_bootstrapped_demos=self.max_bootstrapped_demos,
            max_labeled_demos=self.max_labeled_demos,
            max_rounds=self.max_rounds,
            num_candidate_programs=self.num_candidate_programs,
            num_threads=self.num_threads,
        )

    # NOTE: ideally, optimize should be agnostic of program implementation.
    # NOTE: it should get an optimization config.
    def optimize(
        self,
        program: InferRetrieveRank,
        train_examples: list[dspy.Example],
        validation_examples: list[dspy.Example],
    ) -> dspy.Module:
        # Freeze Rank module if unoptimized or skipped
        if not self.rank_compile or program.rank_skip:
            program.rank.cot._compiled = True
        # Freeze Infer module if unoptimized
        if not self.infer_compile:
            program.infer_retrieve.infer.cot._compiled = True

        # Create Teacher, set LMs
        teacher = program.deepcopy()
        teacher.infer_retrieve.infer.cot.lm = self.modules_to_lms[
            "infer_retrieve.infer"
        ]["teacher"]
        teacher.rank.cot.lm = self.modules_to_lms["rank"]["teacher"]

        # Create compiler
        compiler = self.create_compiler(
            self.rank_compile_metric,
        )

        # Compile
        program = compiler.compile(
            program,
            teacher=teacher,
            trainset=train_examples,
            valset=validation_examples,
            restrict=range(20),
        )

        # Keep track of which modules were compiled
        # NOTE: DSPy should do this internally
        if self.infer_compile:
            program.infer_retrieve.infer.cot._compiled = True
        if self.rank_compile:
            program.rank.cot._compiled = True

        return program


class LeftToRightOptimizer1(LeftToRightOptimizer):
    def optimize(
        self,
        program: InferRetrieveRank,
        train_examples: list[dspy.Example],
        validation_examples: list[dspy.Example],
    ) -> dspy.Module:
        train_examples_1 = train_examples
        train_examples_2 = train_examples
        # First round
        if self.infer_compile:
            # Create first-round teacher
            teacher = program.deepcopy()
            teacher.infer_retrieve.infer.cot.lm = self.modules_to_lms[
                "infer_retrieve.infer"
            ]["teacher"]

            # No ranking in first-round
            # NOTE: we may want to actually optimize Infer with an (un)optimized version of Ranking in the loop.
            rank_skipped = program.rank_skip
            teacher.rank_skip = True
            program.rank_skip = True
            teacher.rank.cot._compiled = True
            program.rank.cot._compiled = True

            # create compiler
            infer_compiler = self.create_infer_compiler(
                self.infer_compile_metric,
            )

            # compile
            program = infer_compiler.compile(
                program,
                teacher=teacher,
                trainset=train_examples_1,
                valset=validation_examples,
                restrict=range(20),
            )

            # Set back Rank module to what it originally was.
            program.rank_skip = rank_skipped

            # Freeze module we just compiled
            # NOTE: this may not be working actually.
            program.infer_retrieve.infer.cot._compiled = True
            program.rank.cot._compiled = False
            program._compiled = False

        # Second round
        if self.rank_compile and not program.rank_skip:
            # Create second-round teacher
            teacher = program.deepcopy()
            teacher.rank.cot.lm = self.modules_to_lms["rank"]["teacher"]

            rank_compiler = self.create_rank_compiler(self.rank_compile_metric)

            # compile
            program = rank_compiler.compile(
                program,
                teacher=teacher,
                trainset=train_examples_2,
                valset=validation_examples,
                restrict=range(20),
            )

            program.rank.cot._compiled = True

        return program

    def create_infer_compiler(self, metric):
        return self.create_compiler(metric)

    def create_rank_compiler(self, metric):
        return self.create_compiler(metric)

    # NOTE: make this better via kwarg passing and optimization config
    def create_compiler(self, metric):
        return BootstrapFewShotWithRandomSearch(
            metric=metric,
            max_bootstrapped_demos=self.max_bootstrapped_demos,
            max_labeled_demos=self.max_labeled_demos,
            max_rounds=self.max_rounds,
            num_candidate_programs=self.num_candidate_programs,
            num_threads=self.num_threads,
            only_reset_uncompiled=True,  # This allows different demonstrations for Infer and Rank module.
        )


class LeftToRightOptimizer2(LeftToRightOptimizer):
    def optimize(
        self,
        program: InferRetrieveRank,
        train_examples: list[dspy.Example],
        validation_examples: list[dspy.Example],
    ) -> dspy.Module:
        train_examples_1 = train_examples[: floor(len(train_examples) / 2)]
        train_examples_2 = train_examples[floor(len(train_examples) / 2) :]
        # First round
        if self.infer_compile:
            # Create first-round teacher
            teacher = program.deepcopy()
            teacher.infer_retrieve.infer.cot.lm = self.modules_to_lms[
                "infer_retrieve.infer"
            ]["teacher"]

            # No ranking in first-round
            # NOTE: we may want to actually optimize Infer with an (un)optimized version of Ranking in the loop.
            rank_skipped = program.rank_skip
            teacher.rank_skip = True
            program.rank_skip = True
            teacher.rank.cot._compiled = True
            program.rank.cot._compiled = True

            # create compiler
            infer_compiler = self.create_infer_compiler(
                self.infer_compile_metric,
            )

            # compile
            program = infer_compiler.compile(
                program,
                teacher=teacher,
                trainset=train_examples_1,
                valset=validation_examples,
                restrict=range(20),
            )

            # Set back Rank module to what it originally was.
            program.rank_skip = rank_skipped

            # Freeze module we just compiled
            # NOTE: this may not be working actually.
            program.infer_retrieve.infer.cot._compiled = True
            program.rank.cot._compiled = False
            program._compiled = False

        # Second round
        if self.rank_compile and not program.rank_skip:
            # Create second-round teacher
            teacher = program.deepcopy()
            teacher.rank.cot.lm = self.modules_to_lms["rank"]["teacher"]

            rank_compiler = self.create_rank_compiler(self.rank_compile_metric)

            # compile
            program = rank_compiler.compile(
                program,
                teacher=teacher,
                trainset=train_examples_2,
                valset=validation_examples,
                restrict=range(20),
            )

            program.rank.cot._compiled = True

        return program

    def create_infer_compiler(self, metric):
        return self.create_compiler(metric)

    def create_rank_compiler(self, metric):
        return self.create_compiler(metric)

    # NOTE: make this better via kwarg passing and optimization config
    def create_compiler(self, metric):
        return BootstrapFewShotWithRandomSearch(
            metric=metric,
            max_bootstrapped_demos=self.max_bootstrapped_demos,
            max_labeled_demos=self.max_labeled_demos,
            max_rounds=self.max_rounds,
            num_candidate_programs=self.num_candidate_programs,
            num_threads=self.num_threads,
            only_reset_uncompiled=True,  # This allows different demonstrations for Infer and Rank module.
        )


class LeftToRightOptimizer3(LeftToRightOptimizer2):
    def create_infer_compiler(self, metric):
        return BootstrapFewShotWithRandomSearch(
            metric=metric,
            max_bootstrapped_demos=4,
            max_labeled_demos=self.max_labeled_demos,
            max_rounds=self.max_rounds,
            num_candidate_programs=self.num_candidate_programs,
            num_threads=self.num_threads,
            only_reset_uncompiled=True,  # This allows different demonstrations for Infer and Rank module.
        )


class LeftToRightOptimizer4(LeftToRightOptimizer2):
    def create_infer_compiler(self, metric):
        return BootstrapFewShotWithRandomSearch(
            metric=metric,
            max_bootstrapped_demos=8,
            max_labeled_demos=self.max_labeled_demos,
            max_rounds=self.max_rounds,
            num_candidate_programs=self.num_candidate_programs,
            num_threads=self.num_threads,
            only_reset_uncompiled=True,  # This allows different demonstrations for Infer and Rank module.
        )


supported_optimizers = {
    "end-to-end": End2EndOptimizer,
    "left-to-right": LeftToRightOptimizer,
    "left-to-right1": LeftToRightOptimizer1,
    "left-to-right2": LeftToRightOptimizer2,
    "left-to-right3": LeftToRightOptimizer3,
    "left-to-right4": LeftToRightOptimizer4,
}


# def bayesian_optimizer_data(default_program, trainset, devset, test_name, dataset_name, kwargs):
#     eval_kwargs = dict(num_threads=10, display_progress=True, display_table=0)
#     teleprompter = BayesianSignatureOptimizer(prompt_model=kwargs["prompt_model"], task_model=kwargs["task_model"], metric=kwargs["metric"], n=10, init_temperature=1.0)
#     compiled_program = teleprompter.compile(default_program, devset=trainset, optuna_trials_num=1, max_bootstrapped_demos=2, max_labeled_demos=5, eval_kwargs=eval_kwargs)

#     return compiled_program
