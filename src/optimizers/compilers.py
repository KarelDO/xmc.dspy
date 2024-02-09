from dspy.teleprompt import (
    BootstrapFewShot,
    BootstrapFewShotWithRandomSearch,
)

from .config import CompilerConfig

supported_compilers = {
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
