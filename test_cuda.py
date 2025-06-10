import torch
print("CUDA Available:", torch.cuda.is_available())
print("PyTorch Version:", torch.__version__)
if torch.cuda.is_available():
    print("CUDA Version:", torch.version.cuda)
    print("Device Count:", torch.cuda.device_count())
    print("Current Device:", torch.cuda.current_device())
    print("Device Name:", torch.cuda.get_device_name(0))
    
    # Test if we can create a tensor on GPU
    try:
        x = torch.randn(1, device='cuda')
        print("✓ Successfully created tensor on GPU")
        del x
        torch.cuda.empty_cache()
        print("✓ GPU memory cleared")
    except Exception as e:
        print("✗ Error creating tensor on GPU:", str(e))
else:
    print("CUDA not available")
