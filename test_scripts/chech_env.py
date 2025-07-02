import torch
print("torch disponible:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())
print("Versión de CUDA (compilada con):", torch.version.cuda)
print("Versión de CUDA (runtime):", torch.cuda.get_device_properties(0).major, ".", torch.cuda.get_device_properties(0).minor)
