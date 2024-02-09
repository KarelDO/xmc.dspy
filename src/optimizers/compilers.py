from dspy.teleprompt import (
    BootstrapFewShot,
    BootstrapFewShotWithRandomSearch,
)

from .config import CompilerConfig

supported_compilers = {
    "bootstrap-few-shot": BootstrapFewShot,
    "bootstrap-few-shot-with-random-search": BootstrapFewShotWithRandomSearch,
}


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
