"""Run only the provenance-approved repair deployment export."""
from native_covariate_experiment_repair1 import main
if __name__ == "__main__":
    main(default_phase="deploy")
