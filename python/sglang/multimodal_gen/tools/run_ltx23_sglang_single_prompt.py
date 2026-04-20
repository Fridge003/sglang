from __future__ import annotations

import argparse

from sglang.multimodal_gen import DiffGenerator
from sglang.multimodal_gen.configs.sample.sampling_params import generate_request_id
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.utils import FlexibleArgumentParser


def _read_prompt(args: argparse.Namespace) -> str:
    if bool(args.single_prompt) == bool(args.single_prompt_file):
        raise ValueError(
            "Exactly one of --single-prompt or --single-prompt-file must be provided."
        )
    if args.single_prompt is not None:
        return args.single_prompt
    with open(args.single_prompt_file, encoding="utf-8") as f:
        return f.read().strip()


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = FlexibleArgumentParser(
        description=(
            "Run SGLang LTX-2.3 generation while treating a prompt file as one "
            "single prompt string instead of one prompt per line."
        )
    )
    parser.add_argument("--single-prompt")
    parser.add_argument("--single-prompt-file")
    parser.add_argument("--sample-output-path", required=True)
    parser.add_argument("--sample-width", type=int, default=768)
    parser.add_argument("--sample-height", type=int, default=512)
    parser.add_argument("--sample-num-frames", type=int, default=121)
    parser.add_argument("--sample-seed", type=int, default=10)
    parser = ServerArgs.add_cli_args(parser)
    return parser.parse_known_args()


def main() -> None:
    args, unknown_args = parse_args()
    prompt = _read_prompt(args)
    server_args = ServerArgs.from_cli_args(args, unknown_args)
    generator = DiffGenerator.from_pretrained(
        model_path=server_args.model_path,
        server_args=server_args,
        local_mode=True,
    )
    generator.generate(
        sampling_params_kwargs={
            "prompt": prompt,
            "output_path": args.sample_output_path,
            "width": args.sample_width,
            "height": args.sample_height,
            "num_frames": args.sample_num_frames,
            "seed": args.sample_seed,
            "request_id": generate_request_id(),
        }
    )


if __name__ == "__main__":
    main()
