"""
Publish diffusion CI ground-truth images to sglang-bot/sglang-ci-data
via the GitHub API (same pattern as publish_traces.py).
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Reuse GitHub API helpers from publish_traces.
# Support both direct script execution and package-style imports.
if __package__:
    from ..publish_traces import (
        create_blobs,
        create_commit,
        get_branch_sha,
        get_tree_sha,
        is_permission_error,
        is_rate_limit_error,
        make_github_request,
        update_branch_ref,
        verify_token_permissions,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from publish_traces import (
        create_blobs,
        create_commit,
        get_branch_sha,
        get_tree_sha,
        is_permission_error,
        is_rate_limit_error,
        make_github_request,
        update_branch_ref,
        verify_token_permissions,
    )

REPO_OWNER = "sglang-bot"
REPO_NAME = "sglang-ci-data"
BRANCH = "main"
TARGET_DIR = "diffusion-ci/consistency_gt"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def collect_images(source_dir):
    """Collect image files from source_dir and return list of (repo_path, content) tuples."""
    files = []
    for entry in sorted(os.listdir(source_dir)):
        ext = os.path.splitext(entry)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue
        full_path = os.path.join(source_dir, entry)
        if not os.path.isfile(full_path):
            continue
        with open(full_path, "rb") as f:
            content = f.read()
        repo_path = f"{TARGET_DIR}/{entry}"
        files.append((repo_path, content))
    return files


def get_recursive_tree(repo_owner, repo_name, tree_sha, token):
    """Get the full recursive tree listing for a commit."""
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/trees/{tree_sha}?recursive=1"
    response = make_github_request(url, token)
    data = json.loads(response)
    return data.get("tree", [])


def build_tree_without_base(repo_owner, repo_name, tree_items, token, max_retries=3):
    """Create a new tree from scratch (no base_tree), using full list of items."""
    import time

    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/trees"
    data = {"tree": tree_items}

    for attempt in range(max_retries):
        try:
            response = make_github_request(url, token, method="POST", data=data)
            return json.loads(response)["sha"]
        except Exception as e:
            if is_rate_limit_error(e):
                raise
            if attempt < max_retries - 1:
                wait_time = 2**attempt
                print(
                    f"Tree creation failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                raise


def publish(source_dir):
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN environment variable not set")
        sys.exit(1)

    files_to_upload = collect_images(source_dir)
    if not files_to_upload:
        print(f"No image files found in {source_dir}")
        return

    print(
        f"Found {len(files_to_upload)} image(s) to upload to {REPO_OWNER}/{REPO_NAME}/{TARGET_DIR}"
    )

    # Verify token
    perm = verify_token_permissions(REPO_OWNER, REPO_NAME, token)
    if perm == "rate_limited":
        print("GitHub API rate-limited, skipping upload.")
        return
    if not perm:
        print("Token permission verification failed.")
        sys.exit(1)

    # Create blobs for new images
    try:
        new_tree_items = create_blobs(REPO_OWNER, REPO_NAME, files_to_upload, token)
    except Exception as e:
        if is_rate_limit_error(e):
            print("Rate-limited during blob creation, skipping.")
            return
        if is_permission_error(e):
            print(
                f"ERROR: Token lacks write permission to {REPO_OWNER}/{REPO_NAME}. "
                "Update GH_PAT_FOR_NIGHTLY_CI_DATA with a token that has contents:write."
            )
            sys.exit(1)
        raise

    # Commit with retry (handle concurrent pushes)
    max_retries = 5
    for attempt in range(max_retries):
        try:
            branch_sha = get_branch_sha(REPO_OWNER, REPO_NAME, BRANCH, token)
            tree_sha = get_tree_sha(REPO_OWNER, REPO_NAME, branch_sha, token)

            # Get the full recursive tree, keep everything outside TARGET_DIR,
            # then add our new blob entries.  This avoids the base_tree bug
            # where GitHub rejects existing sub-tree SHAs as invalid blobs.
            existing_tree = get_recursive_tree(
                REPO_OWNER, REPO_NAME, tree_sha, token
            )
            target_prefix = TARGET_DIR + "/"
            kept_items = [
                {
                    "path": item["path"],
                    "mode": item["mode"],
                    "type": item["type"],
                    "sha": item["sha"],
                }
                for item in existing_tree
                if item["type"] == "blob"
                and not item["path"].startswith(target_prefix)
            ]
            all_tree_items = kept_items + new_tree_items
            print(
                f"Building tree: {len(kept_items)} existing blobs + {len(new_tree_items)} new GT images"
            )

            new_tree_sha = build_tree_without_base(
                REPO_OWNER, REPO_NAME, all_tree_items, token
            )
            commit_msg = f"diffusion-ci: update consistency_gt images ({len(files_to_upload)} files) [automated]"
            commit_sha = create_commit(
                REPO_OWNER, REPO_NAME, new_tree_sha, branch_sha, commit_msg, token
            )
            update_branch_ref(REPO_OWNER, REPO_NAME, BRANCH, commit_sha, token)
            print(
                f"Successfully pushed {len(files_to_upload)} images (commit {commit_sha[:10]})"
            )
            return
        except Exception as e:
            if is_rate_limit_error(e):
                print("Rate-limited, skipping.")
                return
            if is_permission_error(e):
                print(f"ERROR: permission denied to {REPO_OWNER}/{REPO_NAME}")
                sys.exit(1)

            retryable = False
            if hasattr(e, "error_body"):
                if "Update is not a fast forward" in e.error_body:
                    retryable = True
                elif "Object does not exist" in e.error_body:
                    retryable = True

            from urllib.error import HTTPError

            if isinstance(e, HTTPError) and e.code in [422, 500, 502, 503, 504]:
                retryable = True

            if retryable and attempt < max_retries - 1:
                import time

                wait = 2**attempt
                print(
                    f"Attempt {attempt + 1}/{max_retries} failed, retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                print(f"Failed after {attempt + 1} attempts: {e}")
                raise


def main():
    parser = argparse.ArgumentParser(
        description="Publish diffusion GT images to GitHub"
    )
    parser.add_argument(
        "--source-dir", required=True, help="Directory containing GT images"
    )
    args = parser.parse_args()
    publish(args.source_dir)


if __name__ == "__main__":
    main()
