import torch

print("torch version: ", torch.__version__)
print("cuda is available: ", torch.cuda.is_available())
num_of_GPUs = torch.cuda.device_count()
print("the number of GPUs: ", num_of_GPUs)
if num_of_GPUs >= 1:
    current_device_idx = torch.cuda.current_device()
    print("current gpu: ", torch.cuda.get_device_name(current_device_idx))


