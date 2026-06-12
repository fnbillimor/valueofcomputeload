Files in this package
- ec2_power_common.py
- load_instances.py
- ec2_fill_prices.py
- ec2_source_catalog.py
- ec2_gpu_groups_and_tdp.py
- ec2_cpu_groups_and_tdp.py
- ec2_compute_power_and_value_split.py
- ec2_expand_power_output.py
- run_ec2_power_pipeline.py

How to use
1. Put the three instance CSVs into:
   <PROJECT_ROOT>\data\original\instances\
2. Put all Python files into:
   <PROJECT_ROOT>\scripts\
3. Run:
   python run_ec2_power_pipeline.py

Main outputs
- aws_gpu_cpu_ram_ssd_power_and_value_long.csv
- aws_gpu_cpu_ram_ssd_power_and_value_wide.csv
- aws_gpu_cpu_ram_ssd_power_and_value_wide_expanded.csv
- ec2_unresolved_power_mappings_review.csv

User-editable scenario settings
Edit SCENARIO_CONFIG in ec2_power_common.py.
The five cases are: idle, lo, mid, hi, custom.
Each case lets you set:
- cpu_utilization
- gpu_utilization
- ssd_state
- ram_state

Model notes
- RAM uses 0.19 W/GB idle and 0.54 W/GB active.
- RAM is set to active in all default cases.
- Other components power = 0.2 * (CPU + RAM + SSD).
- SSD power is based on enterprise NVMe proxy drives for local SSD instances.
- EBS-only AWS instances are assigned zero host-local SSD power.
