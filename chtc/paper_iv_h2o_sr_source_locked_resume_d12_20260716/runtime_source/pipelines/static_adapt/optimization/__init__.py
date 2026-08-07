"""Canonical staged-ADAPT optimization helpers.

The public names are resolved lazily so ``python -m
pipelines.static_adapt.optimization.phase3_policy_optuna`` can execute without
the package initializer pre-importing the target module.
"""

__all__ = [
    "AlgorithmPolicy",
    "BenchmarkResult",
    "GlobalObjectiveConfig",
    "HamiltonianBenchmarkSpec",
    "InnerOptimizerPolicy",
    "PoolPolicy",
    "ProblemFeatureVector",
    "SizeScaledBudget",
    "StaticObjectiveWeights",
    "StaticScaffoldPolicy",
    "aggregate_global_score",
    "apply_policy_to_pipeline_args",
    "build_parser",
    "build_compile_command",
    "build_static_command",
    "default_trial_params",
    "default_static_benchmark_suite",
    "filter_static_benchmark_suite",
    "main",
    "normalized_static_score",
    "objective_global_agnostic",
    "objective_oracle",
    "oracle_gap",
    "policy_to_cli_args",
    "run_optuna_study",
    "run_oracle_grid",
    "run_static_benchmark",
    "sample_policy_from_trial",
]


def __getattr__(name: str):
    if name in __all__:
        from . import phase3_policy_optuna

        return getattr(phase3_policy_optuna, name)
    raise AttributeError(name)
