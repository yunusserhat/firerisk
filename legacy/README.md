The original user-provided `firerisk.py` is preserved here for reference.
It is not the package entry point. Moving it avoids shadowing `src/firerisk`
when invoking `python -m firerisk`. Its original training behavior is not used
by the reproducible benchmark.
