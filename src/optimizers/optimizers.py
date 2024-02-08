from abc import abstractmethod, ABC
from dspy import Example, Module

from .config import OptimizerConfig
from .compilers import get_compiler


class IReRaOptimizer(ABC):
    def __init__(self, config: OptimizerConfig):
        self.config = config

    @abstractmethod
    def optimize(
        self,
        program: Module,
        train_examples: list[Example],
        train_examples_with_label: list[Example],
        val_examples: list[Example],
    ) -> Module:
        raise NotImplementedError()


class End2EndOptimizer(IReRaOptimizer):
    def _prepare_teacher(self, program: Module) -> Module:
        teacher = self._prepare_student(program)
        # set LMs
        teacher.infer_retrieve.infer.cot.lm = self.config.infer_teacher_model
        teacher.rank.cot.lm = self.config.rank_teacher_model
        return teacher

    def _prepare_student(self, program: Module) -> Module:
        student = program.deepcopy()
        # Freeze Rank module if unoptimized or skipped
        if not self.config.rank_compile or student.rank_skip:
            student.rank.cot._compiled = True

        # Freeze Infer module if unoptimized
        if not self.config.infer_compile:
            student.infer_retrieve.infer.cot._compiled = True
        return student

    def _cleanup_student(self, program: Module) -> None:
        if self.config.infer_compile:
            program.infer_retrieve.infer.cot._compiled = True
        if self.config.rank_compile:
            program.rank.cot._compiled = True

    def optimize(
        self,
        program: Module,
        train_examples: list[Example],
        train_examples_with_label: list[Example],
        val_examples: list[Example],
    ) -> Module:
        student = self._prepare_student(program)
        teacher = self._prepare_teacher(program)

        # the end-to-end optimizer will use the infer compiler kwargs
        compiler_kwargs = self.config.infer_compile_config
        compiler = get_compiler(compiler_kwargs)

        # compile
        student_compiled = compiler.compile(
            student, teacher=teacher, trainset=train_examples, valset=val_examples
        )

        self._cleanup_student(student_compiled)
        return student_compiled


class Left2RightOptimizer(IReRaOptimizer):
    def __init__(self, config: OptimizerConfig):
        super().__init__(config)

        if not self.config.infer_compile or not self.config.rank_compile:
            raise ValueError(
                "Left2RightOptimizer expects both 'infer_compile' and 'rank_compile' to be True."
            )

    def _prepare_infer_student(self, program: Module) -> Module:
        student = program.deepcopy()
        # no ranking in infer round
        student.rank_skip = True
        student.rank.cot._compiled = True
        return student

    def _prepare_infer_teacher(self, program: Module) -> Module:
        teacher = self._prepare_infer_student(program)
        # set lm
        teacher.infer_retrieve.infer.cot.lm = self.config.infer_teacher_model
        return teacher

    def _prepare_rank_student(self, program: Module) -> Module:
        student = program.deepcopy()
        student.rank_skip = False
        student.infer_retrieve.infer.cot._compiled = True
        student.rank.cot._compiled = False
        student._compiled = False
        return student

    def _prepare_rank_teacher(self, program: Module) -> Module:
        teacher = self._prepare_rank_student(program)
        # set lm
        teacher.rank.cot.lm = self.config.rank_teacher_model
        return teacher

    def _cleanup_student(self, program: Module) -> None:
        program.rank.cot._compiled = True

    def optimize(
        self,
        program: Module,
        train_examples: list[Example],
        train_examples_with_label: list[Example],
        val_examples: list[Example],
    ) -> Module:
        if program.rank_skip:
            raise ValueError(
                "Left2RightOptimizer expects 'program.rank_skip' to be False."
            )
        # first round prepare
        student_infer = self._prepare_infer_student(program)
        teacher_infer = self._prepare_infer_teacher(program)

        compiler_infer_kwargs = self.config.infer_compile_config
        compiler_infer = get_compiler(compiler_infer_kwargs)
        compiler_infer.only_reset_uncompiled = True

        # compile first round
        student_infer_compiled = compiler_infer.compile(
            student_infer,
            teacher=teacher_infer,
            trainset=train_examples
            if not self.config.infer_hint
            else train_examples_with_label,
            valset=val_examples,
        )

        # second round prepare
        student_rank = self._prepare_rank_student(student_infer_compiled)
        teacher_rank = self._prepare_rank_teacher(student_infer_compiled)

        compiler_rank_kwargs = self.config.rank_compile_config
        compiler_rank = get_compiler(compiler_rank_kwargs)
        compiler_rank.only_reset_uncompiled = True

        # compile first round
        student_rank_compiled = compiler_rank.compile(
            student_rank,
            teacher=teacher_rank,
            trainset=train_examples,
            valset=val_examples,
        )

        self._cleanup_student(student_rank_compiled)
        return student_rank_compiled


supported_optimizers = {
    "left-to-right": Left2RightOptimizer,
    "end-to-end": End2EndOptimizer,
}


def get_optimizer(config: OptimizerConfig):
    optimizer_name = config.optimizer_name
    optimizer = supported_optimizers[optimizer_name]

    return optimizer(config)
