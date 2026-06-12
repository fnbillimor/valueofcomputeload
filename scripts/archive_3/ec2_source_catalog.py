"""
Source catalog used by the EC2 CPU/GPU power suite.

Purpose:
- documents where TDP/source assumptions may come from
- allows relaxed CPU sourcing beyond official vendor datasheets
- keeps one explicit file listing all intended source classes
"""

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
        "used_for": "Processor family identification and AWS custom-part context",
    },
    {
        "source_group": "benchmark_reference",
        "source_name": "CPU-Monkey",
        "source_type": "benchmark_proxy",
        "used_for": "CPU package characteristics where direct datasheets are missing",
    },
    {
        "source_group": "benchmark_reference",
        "source_name": "PassMark / cpubenchmark.net",
        "source_type": "benchmark_proxy",
        "used_for": "CPU family comparison and closest-SKU inference",
    },
    {
        "source_group": "benchmark_reference",
        "source_name": "TechPowerUp CPU database",
        "source_type": "benchmark_proxy",
        "used_for": "Cross-checking TDP/core/thread characteristics",
    },
    {
        "source_group": "benchmark_reference",
        "source_name": "Notebookcheck CPU pages",
        "source_type": "benchmark_proxy",
        "used_for": "Secondary cross-checking for server/workstation CPU characteristics",
    },
    {
        "source_group": "benchmark_reference",
        "source_name": "WikiChip",
        "source_type": "reference_proxy",
        "used_for": "Architecture family cross-checks and lineage notes",
    },
    {
        "source_group": "benchmark_reference",
        "source_name": "ServeTheHome CPU reference articles",
        "source_type": "reference_proxy",
        "used_for": "Server CPU family context and package positioning",
    },
]

if __name__ == "__main__":
    import pandas as pd
    from ec2_power_common import output_path

    df = pd.DataFrame(SOURCE_CATALOG)
    output_file = output_path("ec2_power_source_catalog.csv")
    df.to_csv(output_file, index=False)

    print(f"Saved → {output_file}")
    print(df.to_string(index=False))
