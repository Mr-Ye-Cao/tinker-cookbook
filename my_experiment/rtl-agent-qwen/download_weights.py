import tinker
import requests
import os
import tarfile
import sys

# Get tinker path from env var (must use sampler_path, not state_path)
tinker_path = os.environ.get("TINKER_PATH")
if not tinker_path:
    print("Error: TINKER_PATH environment variable is required")
    print("Example: export TINKER_PATH='tinker://run-id:train:0/sampler_weights/final'")
    sys.exit(1)

output_path = os.environ.get("OUTPUT_PATH", "/home/ubuntu/peter/weights/downloaded_model")

# Create the REST client
print("Creating REST client...")
service_client = tinker.ServiceClient()
rest_client = service_client.create_rest_client()

# Get the signed download URL
print(f"Getting download URL for: {tinker_path}")
response = rest_client.get_checkpoint_archive_url_from_tinker_path(tinker_path).result()

print(f"Download URL expires: {response.expires}")

# Download the archive
os.makedirs(output_path, exist_ok=True)
archive_path = f"{output_path}/model-checkpoint.tar"

print(f"Downloading to: {archive_path}")
with requests.get(response.url, stream=True) as r:
    r.raise_for_status()
    total_size = int(r.headers.get('content-length', 0))
    downloaded = 0
    with open(archive_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = (downloaded / total_size) * 100
                print(f"\rProgress: {pct:.1f}% ({downloaded / 1024 / 1024:.1f} MB)", end="", flush=True)

print(f"\nDownloaded to: {archive_path}")

# Extract (plain tar, not gzipped)
print(f"Extracting to: {output_path}")
with tarfile.open(archive_path, "r") as tar:
    tar.extractall(output_path)

# Remove the archive after extraction
os.remove(archive_path)

print(f"Extraction complete!")
print(f"Files in {output_path}:")
for item in os.listdir(output_path):
    print(f"  - {item}")
