"""Export the frozen round10 selected forecast after May31; no live forecast."""
from native_covariate_experiment import main

if __name__ == "__main__":
    main(default_phase="deploy")
