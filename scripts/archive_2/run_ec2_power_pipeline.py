from pathlib import Path
import runpy
import sys

BASE_DIR = Path(
    r"C:\Users\FarhadBillimoria\OneDrive - Aurora Energy Research\Documents\Personal\Endog_demand_AI\pyai"
)

# If your scripts are in ...\pyai\scripts, use this:
SCRIPTS_DIR = BASE_DIR / "scripts"

PIPELINE_FILES = [
    "load_instances.py",
    "ec2_fill_prices.py",
    "ec2_source_catalog.py",
    "ec2_gpu_groups_and_tdp.py",
    "ec2_cpu_groups_and_tdp.py",
    "ec2_compute_power_and_value_split.py",
    "ec2_expand_power_output.py",
]


def main():
    if not SCRIPTS_DIR.exists():
        raise FileNotFoundError(f"Scripts directory not found: {SCRIPTS_DIR}")

    # Make sure imports like `import ec2_power_common` work
    scripts_dir_str = str(SCRIPTS_DIR.resolve())
    if scripts_dir_str not in sys.path:
        sys.path.insert(0, scripts_dir_str)

    for file_name in PIPELINE_FILES:
        script_path = SCRIPTS_DIR / file_name

        if not script_path.exists():
            raise FileNotFoundError(f"Pipeline script not found: {script_path}")

        print(f"\n=== Running: {script_path.name} ===")
        runpy.run_path(str(script_path), run_name="__main__")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()