"""Reference catalog for the EC2 power pipeline."""

SOURCE_CATALOG = [
    {
        "source_group": "official_gpu_vendor",
        "source_name": "NVIDIA product pages and datasheets",
        "source_type": "official_datasheet_or_vendor_spec",
        "used_for": "GPU TDP values where official specs are available",
    },
    {
        "source_group": "official_cpu_vendor",
        "source_name": "Intel ARK / Intel Xeon product pages",
        "source_type": "official_vendor_spec",
        "used_for": "CPU TDP values for directly identifiable Intel SKUs",
    },
    {
        "source_group": "official_cpu_vendor",
        "source_name": "AMD EPYC product pages / datasheets",
        "source_type": "official_vendor_spec",
        "used_for": "CPU TDP values for directly identifiable AMD SKUs",
    },
    {
        "source_group": "cloud_vendor",
        "source_name": "AWS EC2 instance documentation",
        "source_type": "cloud_vendor_reference",
        "used_for": "Processor family identification and local-NVMe instance-store context",
    },
    {
        "source_group": "ssd_reference",
        "source_name": "TechPowerUp SSD database (intended proxy class)",
        "source_type": "proxy_reference",
        "used_for": "Enterprise NVMe SSD proxy selection for local SSD idle/active power",
    },
    {
        "source_group": "ssd_vendor_crosscheck",
        "source_name": "Samsung PM9A3 / Kioxia CD6 public datasheets",
        "source_type": "proxy_reference_crosscheck",
        "used_for": "Idle and active SSD power values for chosen enterprise NVMe proxies",
    },
    {
        "source_group": "assumption",
        "source_name": "User-specified RAM coefficients and other-components factor",
        "source_type": "model_assumption",
        "used_for": "RAM idle-active W/GB values and f=0.2 uplift for other components",
    },
]

if __name__ == "__main__":
    import pandas as pd
    from ec2_power_common import output_path

    df = pd.DataFrame(SOURCE_CATALOG)
    output_file = output_path("ec2_power_source_catalog.csv")
    df.to_csv(output_file, index=False)
    print(f"Saved → {output_file}")
