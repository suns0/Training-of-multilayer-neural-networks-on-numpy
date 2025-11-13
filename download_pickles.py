import gdown
import os

# Список файлов: имя -> Google Drive ID
files = {
    "cifar10_train_images.pickle": "1a2MH2pVklJIVdfexJ2LXu3d9F7WQpsDL",
    "cifar10_train_labels.pickle": "1_kxZ4fJ_C82xaY6OxOD6GFmE0jP2KpEF",
    "cifar10_test_images.pickle": "1f7ZJ6rwtQ8bIhq15uAewl04I7VdvzNQz",
    "cifar10_test_labels.pickle": "1Zz6owDXO5GH5hQ8_8-Uav7OxzilyEazw"
}

os.makedirs("13_unittest_cifar10_input", exist_ok=True)

for filename, file_id in files.items():
    url = f"https://drive.google.com/uc?id={file_id}"
    output = os.path.join("13_unittest_cifar10_input", filename)
    print(f"Downloading {filename}...")
    gdown.download(url, output, quiet=False)

print("\n✅ All files downloaded to ./13_unittest_cifar10_input/")
