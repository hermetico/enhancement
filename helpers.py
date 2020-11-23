import os
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

norm_mean = torch.Tensor([0.485, 0.456, 0.406])
norm_std = torch.Tensor([0.229, 0.224, 0.225])

def normalize(x):
    return x.add_(-norm_mean[:, None, None].to(x.device)).mul_(1 / norm_std[:, None, None].to(x.device))


def denormalize(x):
    return x.mul_(norm_std[:, None, None].to(x.device)).add_(norm_mean[:, None, None].to(x.device))


def create_output_folder(path, abs_path=True):
    parts = path.split("/")[:-1]
    folder_path = os.path.join(*parts)
    if abs_path:
        folder_path = "/" + folder_path
    os.makedirs(folder_path, exist_ok=True)


def store_result(image: Image,  output_path: str, jpg_quality=95):
    print("Storing result at", output_path)
    image.save(output_path, quality=jpg_quality, subsampling=0)


def remove_padding(original_size, tensor):
    h, w = original_size
    return tensor[:, :, :h, :w]


def compute_padding(size, factor):
    """Ensures size is divisible by factor, and size / factor is divisible by 2"""
    if size % factor == 0 and (size/factor) % 2 == 0:
        p = 0
    else:
        p = 1
        while (size + p) % factor != 0 or ((size + p) / factor) % 2 !=0:
            p += 1
    return p


def inference(network, tensor: torch.Tensor):
    """tensor should be a 4 dimensonal tensor"""
    with torch.no_grad():
        result = network(tensor)
    return result


def clamp(data):
    return torch.clamp(data, min=0, max=1)


def output_transforms(data):
    result = clamp(data).squeeze(0)
    return to_pil_image(result)