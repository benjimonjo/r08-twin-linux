"""Fast-forward updater for the public GitHub checkout, preserving local OCR data."""
import os
import shutil
import subprocess
import tempfile

USER_DATA = ("r08_config.json", "r08_digits.npz")


def _git(root, *args, check=True):
    return subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=check)


def _paths(root):
    result = _git(root, "status", "--porcelain", "--untracked-files=all")
    return [line[3:].strip() for line in result.stdout.splitlines() if line]


def check_for_updates(root):
    root = os.path.realpath(root)
    if not os.path.exists(os.path.join(root, ".git")):
        raise RuntimeError("This copy is not a Git checkout, so it cannot update itself.")
    remote = _git(root, "remote", "get-url", "origin").stdout.strip()
    normalized_remote = remote.lower().replace(":", "/").removesuffix(".git").rstrip("/")
    if "github.com/benjimonjo/r08_twin" not in normalized_remote:
        raise RuntimeError("The origin remote is not the project's GitHub repository.")
    branch = _git(root, "branch", "--show-current").stdout.strip()
    if not branch:
        raise RuntimeError("Cannot update from a detached Git checkout.")
    target = f"origin/{branch}"
    fetched = _git(root, "fetch", "--quiet", "origin",
                    f"+refs/heads/{branch}:refs/remotes/origin/{branch}", check=False)
    if fetched.returncode:
        raise RuntimeError(f"GitHub fetch failed: {(fetched.stderr or fetched.stdout).strip()}")
    if _git(root, "rev-parse", "--verify", target, check=False).returncode:
        raise RuntimeError(f"Git did not create the expected tracking ref {target}.")
    current = _git(root, "rev-parse", "HEAD").stdout.strip()
    latest = _git(root, "rev-parse", target).stdout.strip()
    counts = _git(root, "rev-list", "--left-right", "--count", f"HEAD...{target}").stdout.split()
    ahead, behind = (int(counts[0]), int(counts[1]))
    dirty = _paths(root)
    blocked = [p for p in dirty if p not in USER_DATA]
    staged = _git(root, "diff", "--cached", "--quiet", check=False).returncode != 0
    return {"root": root, "branch": branch, "target": target, "current": current,
            "latest": latest, "ahead": ahead, "behind": behind,
            "dirty": dirty, "blocked": blocked, "staged": staged}


def install_update(info):
    """Fast-forward only when code is clean; keep each user's config and OCR bank."""
    root, target = info["root"], info["target"]
    if info["behind"] <= 0:
        raise RuntimeError("This checkout has no fast-forward update available.")
    if info["ahead"]:
        raise RuntimeError("Local commits differ from GitHub; automatic update would not be a fast-forward.")
    if _git(root, "diff", "--cached", "--quiet", check=False).returncode:
        raise RuntimeError("Commit or unstage local staged changes before updating.")
    dirty = _paths(root)
    blocked = [p for p in dirty if p not in USER_DATA]
    if blocked:
        raise RuntimeError("Commit or save these local source changes before updating: " + ", ".join(blocked[:8]))
    with tempfile.TemporaryDirectory(prefix="r08-update-") as backup:
        saved = {}
        for name in USER_DATA:
            path = os.path.join(root, name)
            if os.path.isfile(path):
                copy = os.path.join(backup, name)
                shutil.copy2(path, copy)
                saved[name] = copy
        try:
            if dirty:
                _git(root, "restore", "--worktree", "--", *dirty)
            merged = _git(root, "merge", "--ff-only", target, check=False)
            if merged.returncode:
                raise RuntimeError((merged.stderr or merged.stdout).strip() or "Git could not fast-forward this checkout.")
        finally:
            for name in USER_DATA:
                path = os.path.join(root, name)
                if name in saved:
                    shutil.copy2(saved[name], path)
                elif os.path.exists(path):
                    os.remove(path)
    return _git(root, "rev-parse", "HEAD").stdout.strip()
