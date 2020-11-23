import torch
import argparse
from torch.nn import functional as F
from torchvision.transforms.functional import to_tensor
from PIL import Image
from functools import partial

from helpers import store_result, create_output_folder, normalize, denormalize, inference, output_transforms
ENHANCER_MODEL = "models/enhancer.pt"

MODE = "bilinear"
ALIGN_CORNERS = True

JPEG_QUALITY = 95

def get_new_dimensions(current_dim, max_size):
    h, w = current_dim
    if h > w:
        nh = max_size
        nw = int(w * (max_size / h))
    else:
        nw = max_size
        nh = int(h * (max_size / w))

    ## round one pixel if not even
    if nw % 2 == 1:
        nw += 1
    if nh % 2 == 1:
        nh += 1
    return nh, nw


def upscaler(output_size):
    def upscaler_(output_size, data):
        return F.interpolate(data, size=output_size, mode=MODE, align_corners=ALIGN_CORNERS)

    return partial(upscaler_, output_size)


def downscaler(current_size, max_side):
    new_size = get_new_dimensions(current_size, max_side)

    def downscaler_(output_size, data):
        return F.interpolate(data, size=output_size, mode=MODE, align_corners=ALIGN_CORNERS)

    return partial(downscaler_, new_size)



def input_transforms(image, with_size, device):
    data = to_tensor(image).to(device)
    big_size = data.shape[-2:]
    downscale = downscaler(big_size, with_size)
    x_n = normalize(data).unsqueeze(0)
    x_nd = downscale(x_n)
    return x_nd, x_n



def enhance_rmu(network, image: Image, size: int, device, half_precision: bool = False) -> Image:
    x_nd, x_n = input_transforms(image, size, device)
    h, w = x_nd.shape[-2:]
    print(f"Running inference at {w}x{h} pixels")
    output_size = x_n.shape[-2:]
    upscale = upscaler(output_size)

    if half_precision:
        _y_hat_nd = inference(network, x_nd.half()).float()
    else:
        _y_hat_nd = inference(network, x_nd)


    bmask_dn = (_y_hat_nd - x_nd).detach()
    bmask_n = upscale(bmask_dn)

    y_hat_n = bmask_n + x_n
    y_hat = denormalize(y_hat_n).cpu()
    result_image = output_transforms(y_hat)
    return result_image


def enhance_straight(network, image: Image, device, half_precision: bool = False, ) -> Image:
    print("straight")
    x_n = normalize(to_tensor(image).to(device)).unsqueeze(0)
    h, w = x_n.shape[-2:]
    print(f"Running inference at {w}x{h} pixels")
    if half_precision:
        _y_hat_n = inference(network, x_n.half()).float()
    else:
        _y_hat_n = inference(network, x_n)

    y_hat = denormalize(_y_hat_n).cpu()

    result_image = output_transforms(y_hat)
    return result_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhancer with Residual Mask Upsaling')
    parser.add_argument("--gpu",  default=None, type=int)
    parser.add_argument("--half", action='store_true', default=False)
    parser.add_argument("--input", "-i", default="examples/original/4663.jpg")
    parser.add_argument("--truth", "-t", default="examples/C/4663.jpg")
    parser.add_argument("--out", "-o", default="out/rmu-4663.jpg")
    parser.add_argument("--i_resolution", default=1024, type=int, help="Max inference resolution")
    parser.add_argument("--f_resolution", default=None, type=int, help="Max final resolution")

    args = parser.parse_args()

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        device = torch.device(torch.cuda.current_device())
        print("Using cuda device:", device)
    else:
        device = torch.device("cpu")

    enhancer = torch.jit.load(ENHANCER_MODEL).to(device)
    enhancer.eval()

    if args.half:
        enhancer = enhancer.half()

    size = args.i_resolution
    image = Image.open(args.input)
    w, h = image.size
    if args.f_resolution is not None:
        nh, nw = get_new_dimensions((h, w), int(args.f_resolution))
        image = image.resize((nw, nh))

    w, h = image.size

    if (w <= size and h <= size):
        edited_image = enhance_straight(enhancer, image, half_precision=args.half, device=device)
    else:
        edited_image = enhance_rmu(enhancer, image, size=size, half_precision=args.half, device=device)
    create_output_folder(args.out, abs_path=False)
    store_result(edited_image, args.out)



