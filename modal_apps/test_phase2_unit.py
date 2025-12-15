import modal
import sys
import os

# Define image with dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "numpy",
        "pandas",
        "scikit-learn",
        "h5py"
    )
)

local_dir = "/Users/suvajitmajumder/multi-turn-reasoning-safety"

# Add local files to image
image = image.add_local_dir(
    local_dir,
    remote_path="/workspace",
    copy=True,  # Copy files into image layer at build time
    ignore=[".venv", ".git", "__pycache__", "representations", "wandb"]
)

# Create app with the fully configured image
app = modal.App("phase2-unit-tests", image=image)

@app.function()
def run_tests():
    import subprocess
    import sys
    import os
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Contents of /: {os.listdir('/')}")
    print(f"Contents of /: {os.listdir('/')}")
    # print(f"Contents of /root: {os.listdir('/root')}")
    
    # Try to find where pkg is
    if os.path.exists("/workspace"):
        print(f"Contents of /workspace: {os.listdir('/workspace')}")
        os.chdir("/workspace")
        sys.path.append("/workspace")
    else:
        print("WARNING: /workspace not found. Trying to find mounted files...")
        # Fallback or exit
        raise FileNotFoundError("/workspace not found in container")
    
    print("Running verification script on Modal...")
    result = subprocess.run(
        [sys.executable, "verify_phase2_implementation.py"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("ERRORS:", result.stderr)
        
    if result.returncode != 0:
        raise Exception("Tests failed!")
    
    print("Tests passed successfully.")

if __name__ == "__main__":
    # This entrypoint allows running `modal run modal_apps/test_phase2_unit.py`
    # from the project root
    with app.run():
        run_tests.remote()
