# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Optional
from dotenv import load_dotenv

from requests.exceptions import HTTPError


def hf_download(repo_id: Optional[str] = None, hf_token: Optional[str] = None) -> None:
    from huggingface_hub import snapshot_download

    os.makedirs(f"checkpoints/{repo_id}", exist_ok=True)

    # if directory is not empty, don't download
    if os.listdir(f"checkpoints/{repo_id}"):
        print(
            f'Directory checkpoints/{repo_id} is not empty, skipping download. First, "rm -rf checkpoints/{repo_id}" if you want to re-download.'
        )
        return

    try:
        snapshot_download(
            repo_id,
            local_dir=f"checkpoints/{repo_id}",
            local_dir_use_symlinks=False,
            token=hf_token,
        )
    except HTTPError as e:
        if e.response.status_code == 401:
            print(
                "You need to pass a valid `--hf_token=...` to download private checkpoints."
            )
        else:
            raise e


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download data from HuggingFace Hub.")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="karpathy/tinyllamas",
        help="Repository ID to download from.",
    )
    parser.add_argument(
        "--hf_token", type=str, default=None, help="HuggingFace API token."
    )

    args = parser.parse_args()

    if args.hf_token is None:
        load_dotenv()
        parser.hf_token = os.getenv("HF_TOKEN")

    hf_download(args.repo_id, args.hf_token)
