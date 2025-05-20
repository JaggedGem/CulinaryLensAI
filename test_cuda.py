import torch
import sys

def check_cuda():
    """Check and display CUDA information"""
    print(f"PyTorch version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        # Get device count
        device_count = torch.cuda.device_count()
        print(f"Number of CUDA devices: {device_count}")
        
        # Get device information
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            print(f"  Device {i}: {device_name}")
            
            # Get device properties
            device_props = torch.cuda.get_device_properties(i)
            print(f"    Total memory: {device_props.total_memory / 1e9:.2f} GB")
            print(f"    CUDA capability: {device_props.major}.{device_props.minor}")
            
        # Test a simple CUDA operation to verify functionality
        try:
            x = torch.rand(1000, 1000).cuda()
            y = torch.rand(1000, 1000).cuda()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            z = torch.matmul(x, y)
            end_event.record()
            
            torch.cuda.synchronize()
            print(f"\nMatrix multiplication test:")
            print(f"  Time: {start_event.elapsed_time(end_event):.2f} ms")
            print(f"  Result sum: {z.sum().item()}")
            print("\nCUDA is working properly!")
            return True
        except Exception as e:
            print(f"\nError during CUDA test: {e}")
            print("CUDA is available but there might be issues with its functionality.")
            return False
    else:
        print("\nCUDA is not available. The training will use CPU, which will be much slower.")
        print("To use GPU acceleration, please install CUDA and the CUDA-enabled version of PyTorch.")
        print("See: https://pytorch.org/get-started/locally/")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("CUDA Availability Test")
    print("=" * 50)
    
    result = check_cuda()
    
    print("\n" + "=" * 50)
    if result:
        print("✅ GPU acceleration is available and working!")
        print("You can train your model with '--device cuda'")
        sys.exit(0)
    else:
        print("❌ GPU acceleration is not available.")
        print("Training will fall back to CPU (much slower)")
        sys.exit(1)
