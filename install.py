import subprocess
import sys

def is_torch_installed():
    """Check if PyTorch is already installed."""
    try:
        import torch
        return True
    except ImportError:
        return False

def is_cuda_available():
    """Check if CUDA is available (requires PyTorch to be installed)."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def install_torch_with_cuda():
    """Install PyTorch with GPU (CUDA) support."""
    print("Installing PyTorch with GPU support...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch>=1.7.0+cu117", "torchvision>=0.8.0+cu117", "torchaudio>=0.7.0",
        "-f", "https://download.pytorch.org/whl/torch_stable.html"
    ])

def install_torch_cpu():
    """Install PyTorch for CPU only."""
    print("Installing PyTorch for CPU...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch>=1.7.0", "torchvision>=0.8.0", "torchaudio>=0.7.0"])

def install_dependencies():
    """Main installation logic."""
    if is_torch_installed():
        if is_cuda_available():
            print("PyTorch with CUDA is already installed. No installation needed.")
        else:
            print("PyTorch is installed but CUDA is not available.")
    else:
        print("PyTorch is not installed. Attempting installation...")
        try:
            # Attempt to install with CUDA first
            install_torch_with_cuda()
            print("Successfully installed PyTorch with CUDA.")
        except subprocess.CalledProcessError:
            print("Failed to install PyTorch with CUDA. Falling back to CPU-only installation.")
            install_torch_cpu()

if __name__ == "__main__":
    install_dependencies()
