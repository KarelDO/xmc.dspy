from dspy.teleprompt import (
    BootstrapFewShot,
    BootstrapFewShotWithRandomSearch,
)

from .config import CompilerConfig


import dsp
from dspy.primitives import Example
from dspy.teleprompt import Teleprompter

# class IterativeBootstrapFewShotGain(Teleprompter):


class BootstrapFewShotGain(BootstrapFewShot):
    # Measure on success over student, not just success.
    def _bootstrap_one_example(self, example, round_idx=0):
        name2traces = self.name2traces
        teacher = self.teacher  # .deepcopy()
        predictor_cache = {}

        try:
            with dsp.settings.context(trace=[], **self.teacher_settings):
                lm = dsp.settings.lm
                lm = (
                    lm.copy(temperature=0.7 + 0.001 * round_idx)
                    if round_idx > 0
                    else lm
                )
                new_settings = dict(lm=lm) if round_idx > 0 else {}

                with dsp.settings.context(**new_settings):
                    for name, predictor in teacher.named_predictors(
                        only_uncompiled=True
                    ):
                        predictor_cache[name] = predictor.demos
                        predictor.demos = [x for x in predictor.demos if x != example]

                    # NOTE: trace should not work like this.
                    teacher_prediction = teacher(**example.inputs())
                    teacher_trace = dsp.settings.trace[
                        : len(teacher.named_predictors(only_uncompiled=True))
                    ]

                    student_prediction = self.student(**example.inputs())
                    student_trace = dsp.settings.trace[
                        len(teacher.named_predictors(only_uncompiled=True)) :
                    ]

                    for name, predictor in teacher.named_predictors(
                        only_uncompiled=True
                    ):
                        predictor.demos = predictor_cache[name]

                    teacher_metric = self.metric(
                        example, teacher_prediction, teacher_trace
                    )
                    student_metric = self.metric(
                        example, student_prediction, student_trace
                    )

                    # NOTE: this can sample trivial cases where student is bad at formatting.
                    # NOTE: we also want to show that each step adds performance. Maybe this is just left-to-right
                    gain = teacher_metric - student_metric

                success = gain > 0
                # print(success, example, prediction)
        except Exception as e:
            success = False
            with self.error_lock:
                self.error_count += 1
                current_error_count = self.error_count
            if current_error_count >= self.max_errors:
                raise e
            print(
                f"Failed to run or to evaluate example {example} with {self.metric} due to {e}."
            )

        if success:
            for step in teacher_trace:
                predictor, inputs, outputs = step

                if "dspy_uuid" in example:
                    demo = Example(
                        augmented=True, dspy_uuid=example.dspy_uuid, **inputs, **outputs
                    )
                else:
                    # TODO: FIXME: This is a hack. RandomSearch will complain for now in this edge case.
                    demo = Example(augmented=True, **inputs, **outputs)

                try:
                    predictor_name = self.predictor2name[id(predictor)]
                except KeyError as e:
                    continue  # FIXME: !

                    # TODO: Look closer into this. It's a bit tricky to reproduce.
                    print(
                        f"Failed to find predictor {predictor} in {self.predictor2name}."
                    )
                    print(
                        "Are you doing this in a notebook (Jupyter)? This might be caused by redefining values by rerunning cells."
                    )
                    print("Try restarting the notebook, or open an issue.")
                    raise KeyError(
                        f"Failed to find predictor {id(predictor)} {predictor} in {self.predictor2name}."
                    ) from e

                name2traces[predictor_name].append(demo)

        return success


# Success metric
# Naive: teacher success
# Less naive: teacher - student success (marginally more costly)
# Even less naive: student validation increase (this is costly; needs to be rerun for every bootstrap)

# New optimizer that directly takes student performance into account. Greedy search.
#
# IterativeBootstrapFewShot
## 1. bootstrap some examples (or just one)
## 2. While condition is not met (combination of tries, max bootstraps, etc.)
### 2.1 add one to student, accept if performance goes up.
### 2.2 also add to teacher, rebootstrap new cases.


supported_compilers = {
    "bootstrap-few-shot-gain": BootstrapFewShotGain,
    "bootstrap-few-shot": BootstrapFewShot,
    "bootstrap-few-shot-with-random-search": BootstrapFewShotWithRandomSearch,
}


def get_compiler(config: CompilerConfig):
    compiler_name = config.compiler_name
    compiler = supported_compilers[compiler_name]

    kwargs = {
        "metric": config.metric,
        "max_bootstrapped_demos": config.max_bootstrapped_demos,
        "max_labeled_demos": config.max_labeled_demos,
    }
    if "random-search" in compiler_name:
        kwargs["num_candidate_programs"] = config.num_candidate_programs
    return compiler(**kwargs)
