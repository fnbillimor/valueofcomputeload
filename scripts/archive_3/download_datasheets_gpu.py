from pathlib import Path
import requests


BASE_DIR = Path(
    r"C:\Users\FarhadBillimoria\OneDrive - Aurora Energy Research\Documents\Personal\Endog_demand_AI\pyai"
)

DATASHEET_DIR = BASE_DIR / "data" / "external" / "gpu_datasheets"
DATASHEET_DIR.mkdir(parents=True, exist_ok=True)


DATASHEETS = {
    # Existing
    "T4.pdf": "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet.pdf",
    "T4G.pdf": "https://d1.awsstatic.com/product-marketing/ec2/NVIDIA_AWS_T4G_DataSheet_FINAL_02_17_2022.pdf",
    "L4.pdf": "https://resources.nvidia.com/en-us-data-center-overview-mc/en-us-data-center-overview/l4-gpu-datasheet",
    "L40S.pdf": "https://www.pny.com/en-eu/File%20Library/Professional/DATASHEET/DATA%20CENTER%20CARDS/PNY-NVIDIA-L40S-Datasheet.pdf",
    "A10.pdf": "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a10/pdf/a10-datasheet.pdf",
    
    "V100.pdf": "https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf",
    "A100.pdf": "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf",
    "H100.pdf": "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/h100/pdf/nvidia-h100-datasheet.pdf",

    # Your additions
    "A10G.pdf": "https://d1.awsstatic.com/product-marketing/ec2/NVIDIA_AWS_A10G_DataSheet_FINAL_02_17_2022.pdf",
    "B200.pdf": "https://www.primeline-solutions.com/media/categories/server/nach-gpu/nvidia-hgx-h200/nvidia-blackwell-b200-datasheet.pdf",
    "B300.pdf": "https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-ultra-datasheet",
    "GB200.pdf": "https://nvdam.widen.net/s/wwnsxrhm2w/blackwell-datasheet-3384703",
}


def download_file(name, url):
    path = DATASHEET_DIR / name

    if path.exists():
        print(f"Already exists: {name}")
        return

    response = requests.get(url, timeout=30)

    if response.status_code == 200:
        with open(path, "wb") as f:
            f.write(response.content)
        print(f"Saved: {name}")
    else:
        print(f"Failed: {name} (status {response.status_code})")


def main():
    for name, url in DATASHEETS.items():
        download_file(name, url)


if __name__ == "__main__":
    main()