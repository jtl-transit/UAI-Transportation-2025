# convert_ckpt_to_safetensors.py
import torch
import argparse
from safetensors.torch import save_file

def convert_ckpt_to_safetensors(src_path, dst_path):
    """Convert PyTorch checkpoint to safetensors format with fp16 precision."""
    ckpt = torch.load(src_path, map_location="cpu", weights_only=True)
    state = ckpt.get("state_dict", ckpt)
    
    # cast float tensors to fp16 to halve size
    for k, v in list(state.items()):
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            state[k] = v.half()
    
    save_file(state, dst_path)
    print(f"Wrote {dst_path} with {len(state)} tensors")

def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch checkpoint to safetensors with fp16 precision")
    parser.add_argument("input", help="Input model path (.ckpt)")
    parser.add_argument("output", help="Output model path (.safetensors)")
    
    args = parser.parse_args()
    convert_ckpt_to_safetensors(args.input, args.output)

if __name__ == "__main__":
    main()