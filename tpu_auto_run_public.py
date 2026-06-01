import os
import subprocess
import time
import sys

# ==============================================================================
# [CONFIGURATION] Customize your TPU target zones and types here.
# Make sure your TRC approved quota matches the chip counts.
# ==============================================================================

# Predefined TPU specifications (Chip counts and versions)
TPU_V5_LITE_64 = {"type": "v5litepod-64", "version": "v2-alpha-tpuv5-lite"}
TPU_V4_32      = {"type": "v4-32",        "version": "v2-alpha-tpuv4"}
TPU_V6E_64     = {"type": "v6e-64",       "version": "v2-alpha-tpuv6e"}

ZONES_CONFIG = [
    {"zone": "europe-west4-a", **TPU_V6E_64,     "spot": True, "name": "auto-tpu-v6e-64-eu"},
    {"zone": "us-central1-a",  **TPU_V5_LITE_64, "spot": True, "name": "auto-tpu-v5lite-64"},
    {"zone": "us-central2-b",  **TPU_V4_32,      "spot": True, "name": "auto-tpu-v4-32"},
]

# Set your local project path. Use "./" for current directory or specify your absolute path.
LOCAL_PATH = "./"  # e.g., "/your/absolute/path/here"
# Buffer time (in seconds) to allow Google Cloud backend to properly release resources after deletion
COOLDOWN_TIME = 600
RUN_COUNT = 1

def run_command(cmd_list):
    """Executes a command and returns success status along with its output"""
    # In Windows, joining a list into a string is often safer when shell=True
    # For Linux/Mac users, it is recommended to set shell=False and pass cmd_list directly.
    cmd_str = ' '.join(cmd_list)
    print(f"-> Executing: {cmd_str}")
    result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr

while True:
    print(f"\n========================================================")
    print(f" TOTAL LOOP RUN COUNT: {RUN_COUNT}")
    print(f"========================================================")
    
    tpu_established = False
    active_config = None

    # 0. First, check if there's already a READY node
    for config in ZONES_CONFIG:
        zone = config["zone"]
        name = config["name"]
        
        print(f"\n[INFO] Checking if TPU already exists and is READY: [{zone}] {name}")
        check_cmd = ["gcloud", "compute", "tpus", "tpu-vm", "describe", name, "--zone", zone, "--format=value(state)"]
        success, stdout, _ = run_command(check_cmd)
        
        if success and "READY" in stdout:
            print(f"[SUCCESS] Found existing READY TPU in {zone}: {name}")
            tpu_established = True
            active_config = config
            break

    # If no READY node exists, attempt to create one
    if not tpu_established:
        for config in ZONES_CONFIG:
            zone = config["zone"]
            accel_type = config["type"]
            version = config["version"]
            name = config["name"]
            
            print(f"\n[INFO] Trying ZONE: [{zone}] | TYPE: [{accel_type}]")
            
            # 1. Clean up any lingering instance (ignore if it fails)
            run_command(["gcloud", "compute", "tpus", "tpu-vm", "delete", name, "--zone", zone, "--quiet"])
            
            # 2. Assemble the TPU creation command
            create_cmd = ["gcloud", "compute", "tpus", "tpu-vm", "create", name, "--zone", zone, "--accelerator-type", accel_type, "--version", version, "--quiet"]
            if config["spot"]:
                create_cmd.append("--spot")
                
            # 3. Attempt creation
            success, _, stderr = run_command(create_cmd)
            if success:
                print(f"[SUCCESS] Created TPU in {zone}: {name}")
                tpu_established = True
                active_config = config
                break
            else:
                print(f"[FAILED] No resources or error in {zone}. moving to next candidate.")
                if stderr:
                    print(f"   Reason: {stderr.strip()}")

                print("[INFO] Waiting 100 seconds for Google Cloud backend to release resources...")
                time.sleep(100)

    # Enter cooldown if failed to secure resources in all zones
    if not tpu_established:
        print(f"\n[WARNING] All zones failed. Waiting {COOLDOWN_TIME} seconds before restarting loop...")
        time.sleep(COOLDOWN_TIME)
        RUN_COUNT += 1
        continue

    # Upon successful resource allocation
    name = active_config["name"]
    zone = active_config["zone"]
    
    print(f"\n========================================================")
    print(f" [SUCCESS] TPU Node is READY!")
    print(f" Zone: {zone}")
    print(f" Name: {name}")
    print(f"========================================================")
    
    ssh_command = f"gcloud compute tpus tpu-vm ssh {name} --zone {zone}"
    # Note: You can also connect via web browser by navigating to Google Cloud Console (Your Project > TPU > SSH)
    print(f"\n[INFO] You can manually connect anytime using this command:")
    print(f"   {ssh_command}")
    print(f"   (Or via web: Google Cloud Console > Your Project > TPU > click 'SSH')\n")
    
    # 1. Temporarily create a local setup script for Linux to be executed remotely
    # In this example, torch, torch-xla, and numpy are installed. Modify as needed.
    setup_script_content = """#!/bin/bash
echo "=== Start Automating TPU Setup ==="
pip install torch==2.9.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch_xla[tpu]==2.9.0 -f https://storage.googleapis.com/libtpu-releases/index.html
pip install "numpy<2.0.0"
echo "=== TPU Setup Completed ==="
"""
    script_filename = "tpu_setup.sh"
    with open(script_filename, "w", newline='\n') as f: # Maintain Linux format (LF)
        f.write(setup_script_content)

    print("[INFO] Transferring setup script to TPU VM via SCP...")
    
    # 2. Copy the file to the remote VM using gcloud compute tpus tpu-vm scp
    # Use os.path.abspath for compatibility with Windows paths
    local_script_path = os.path.abspath(script_filename)
    scp_cmd = ["gcloud", "compute", "tpus", "tpu-vm", "scp", local_script_path, f"{name}:~/", "--zone", zone]
    scp_success, _, scp_err = run_command(scp_cmd)
    
    if scp_success:
        print("[SUCCESS] Script transferred. Executing setup script on remote TPU VM...")
        
        # 3. Execute the copied bash script remotely via SSH (simplifies quote escaping)
        remote_exec_cmd = ["gcloud", "compute", "tpus", "tpu-vm", "ssh", name, "--zone", zone, "--command", "bash ~/tpu_setup.sh"]
        ssh_success, stdout, stderr = run_command(remote_exec_cmd)
        
        if ssh_success:
            print("[SUCCESS] TPU Node setup completed automatically.")
            print(stdout)
        else:
            print(f"[FAILED] TPU setup execution failed:\n{stderr}")
    else:
        print(f"[FAILED] Failed to transfer setup script via SCP:\n{scp_err}")
        
    # Delete the temporary local file
    if os.path.exists(script_filename):
        os.remove(script_filename)

    print("[SUCCESS] TPU Auto-Run script finished.")
    sys.exit(0)
