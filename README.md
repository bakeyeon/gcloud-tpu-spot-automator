# Cloud TPU VM Auto-Provisioner & Environment Setup Automator

A robust Python utility script designed for ML researchers to automate the orchestration of Google Cloud TPU VMs, specifically tailored for **Google TPU Research Cloud (TRC)** users. 

This script continuously hunts for available **Spot TPU resources** across multiple zones, automatically provisions the hardware, and securely initializes the software environment without failing from silent SSH command truncation.

## Key Features
* **Multi-Zone Spot Hunting:** Iterates through predefined zones and configurations to find available TPU slices (v4, v5e/v5lite, v6e) without manual monitoring.
* **State Verification:** Automatically checks if a `READY` node already exists before attempting new creations to prevent redundant provisioning.
* **Robust Software Initialization:** Fixes the common Windows/Linux `gcloud ssh --command` string escaping issue by dynamically packaging and transferring a native Bash script (`tpu_setup.sh`) via SCP.
* **Pre-aligned Version Mapping:** Out-of-the-box support for strict version alignment between `torch==2.9.0` and `torch_xla==2.9.0` alongside `numpy<2.0.0` constraints.

## Prerequisites

This script requires the **Google Cloud CLI (SDK)** installed on your local machine. 

1. **Install Google Cloud CLI:** Follow the official installation guide for your operating system:
   * [Google Cloud CLI Installation Guide](https://cloud.google.com/sdk/docs/install)

2. **Authenticate and Set Project:** Run the following commands in your terminal to log in and set your default project:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID

## Configuration
You can easily customize the zone targets and hardware specifications in the ZONES_CONFIG block within the script:

```Python
ZONES_CONFIG = [
    {"zone": "europe-west4-a", **TPU_V6E_64,     "spot": True, "name": "auto-tpu-v6e-64-eu"},
    {"zone": "us-central1-a",  **TPU_V5_LITE_64, "spot": True, "name": "auto-tpu-v5lite-64"},
]

## Usage
Simply run the Python script on your local machine:

```Bash
python tpu_automator.py

## License
This project is licensed under the MIT License.
