import torch
import argparse
from torch.nn import functional as F
from torchvision.transforms.functional import to_tensor

from PIL import Image
from helpers import store_result, create_output_folder, compute_padding, remove_padding
from helpers import clamp, normalize, denormalize, output_transforms

ENHANCER_MODEL = "models/enhancer.pt"
UPSCALER_MODEL = "models/upscaler.pt"
FACTOR = 6

MODE = "bicubic"
ALIGN_CORNERS = True
JPEG_QUALITY = 95

norm_mean = torch.Tensor([0.485, 0.456, 0.406])
norm_std = torch.Tensor([0.229, 0.224, 0.225])


def enhance(enhancer, upscaler, image: Image, device, factor: int = 5, half_precision: bool = False) -> Image:

    x = to_tensor(image).to(device).unsqueeze(0)
    h, w = x.shape[-2:]  # original sizes
    x_size = (h,w)
    hp, wp = compute_padding(h, factor), compute_padding(w, factor)
    padding = (0, wp, 0, hp)    # pading: left, right, top, bottom
    x_prime = F.pad(x, padding, mode="replicate")
    h_prime, w_prime = x_prime.shape[-2:]



    x_prime_prime_size = (h_prime//factor, w_prime//factor)
    x_prime_n = normalize(x_prime)
    x_prime_prime_n = F.interpolate(x_prime_n, size=x_prime_prime_size, mode=MODE, align_corners=True)
    print(f"Running inference at {x_prime_prime_size[1]}x{x_prime_prime_size[0]} pixels")
    # enhance the input, output is normalized
    with torch.no_grad():
        if half_precision:
            y_hat_prime_prime_n = enhancer(x_prime_prime_n.half())
            guide = x_prime_n.half()
            y_hat_prime_n = F.interpolate(y_hat_prime_prime_n, guide.shape[-2:], mode='nearest')
            y_hat_prime_n = upscaler(torch.cat([guide, y_hat_prime_n], dim=1)).float()
        else:
            y_hat_prime_prime_n = enhancer(x_prime_prime_n)
            guide = x_prime_n
            y_hat_prime_n = F.interpolate(y_hat_prime_prime_n, guide.shape[-2:], mode='nearest')
            y_hat_prime_n = upscaler(torch.cat([guide, y_hat_prime_n], dim=1))

    y_hat_prime = denormalize(y_hat_prime_n)
    y_hat = remove_padding(x_size, y_hat_prime)
    result = clamp(y_hat)

    return output_transforms(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhancer with Deep Joint Upsampler')
    parser.add_argument("--gpu", default=None, type=int)
    parser.add_argument("--half", action='store_true', default=False)
    parser.add_argument("--input", "-i", default="examples/original/4663.jpg")
    parser.add_argument("--truth", "-t", default="examples/C/4663.jpg")
    parser.add_argument("--out", "-o", default="out/dju-4663.jpg")

    args = parser.parse_args()

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        device = torch.device(torch.cuda.current_device())
        print("Using cuda device:", device)
    else:
        device = torch.device("cpu")

    enhancer = torch.jit.load(ENHANCER_MODEL).to(device)
    enhancer.eval()

    upscaler = torch.jit.load(UPSCALER_MODEL).to(device)
    upscaler.eval()

    if args.half:
        enhancer = enhancer.half()
        upscaler = upscaler.half()

    image = Image.open(args.input)
    w, h = image.size

    edited_image = enhance(enhancer, upscaler, image, device=device, half_precision=args.half, factor=FACTOR)
    create_output_folder(args.out, abs_path=False)
    store_result(edited_image, args.out)
