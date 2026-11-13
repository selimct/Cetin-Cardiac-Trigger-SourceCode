"""Discover published model families, variant subdirectories and CV runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

CV_METHODS = ("nested_lopo", "nested_lopo_hpo", "nested_recovery_lopo")
RECOVERY_DIRECTORIES = ("recovery", "frozen_recovery", "lopo_recovery")
PUBLISHED_VARIANT_DIRECTORIES = (*RECOVERY_DIRECTORIES, "delayed_100hz")


def published_model_directory(modelname, root="finished_models"):
    """Return the publication path while retaining a unique declared model name."""
    for kind in ("frozen_recovery", "lopo_recovery", "recovery", "delayed_100hz"):
        suffix = "_" + kind
        if modelname.endswith(suffix) and modelname != suffix:
            return Path(root) / modelname.removesuffix(suffix) / kind
    return Path(root) / modelname


@dataclass(frozen=True)
class ModelArtifact:
    name: str
    directory: Path
    family: str
    method: str | None = None

    def matches(self, name):
        return name in {self.name, self.family, self.family.removesuffix("_cv")}


def discover_models(root, *, variant="best", method=None, names=(), missing_ok=False):
    if method not in (None, *CV_METHODS):
        raise ValueError(f"Unknown CV method: {method}")
    if variant not in {"best", "tuned"}:
        raise ValueError("variant must be best or tuned")
    root = Path(root)
    checkpoint = "best.pth" if variant == "best" else "best_tuned.pth"
    config = "train_config.json" if variant == "best" else "tuning/tuned_config.json"
    candidates = []
    for family in sorted(root.iterdir()) if root.is_dir() else ():
        if not family.is_dir() or family.name.startswith("."):
            continue
        locations = [(family, None)] if method is None else []
        if method is None:
            locations.extend(
                (family / kind, None) for kind in PUBLISHED_VARIANT_DIRECTORIES
            )
        locations.extend(
            (family / value, value)
            for value in (CV_METHODS if method is None else [method])
        )
        for directory, cv_method in locations:
            if (
                not (directory / checkpoint).is_file()
                or not (directory / config).is_file()
            ):
                continue
            cfg = json.loads((directory / config).read_text())
            name = cfg.get("modelname") or family.name
            if (
                Path(name).name != name
                or name in {".", ".."}
                or any(char in name for char in "\t\r\n")
            ):
                raise ValueError(f"Invalid model name in {directory / config}")
            # Base selection must not also select sibling variants. Each keeps
            # a unique declared name across all model families.
            alias = (
                name if directory.name in PUBLISHED_VARIANT_DIRECTORIES else family.name
            )
            artifact = ModelArtifact(name, directory.resolve(), alias, cv_method)
            if not names or any(artifact.matches(value) for value in names):
                candidates.append(artifact)
    for name in names:
        if not missing_ok and not any(
            artifact.matches(name) for artifact in candidates
        ):
            raise ValueError(f"No {variant} model found for {name!r} under {root}")
    selected = []
    for name in sorted({candidate.name for candidate in candidates}):
        matches = [candidate for candidate in candidates if candidate.name == name]
        flat = [candidate for candidate in matches if candidate.method is None]
        if method is None and flat:
            matches = flat
        if len(matches) != 1:
            locations = ", ".join(str(candidate.directory) for candidate in matches)
            raise ValueError(
                f"Ambiguous model {name!r}: {locations}. Select --cv-method or an exact --model-dir."
            )
        selected.append(matches[0])
    return selected


def resolve_model(
    name, *, roots=("models", "finished_models"), variant="best", method=None
):
    # Preserve the old models/NAME default when present, then prefer published
    # copies before falling back to a unique nested training run.
    if method is None:
        for root in roots:
            for directory in dict.fromkeys(
                (published_model_directory(name, root), Path(root) / name)
            ):
                checkpoint = "best.pth" if variant == "best" else "best_tuned.pth"
                config = (
                    "train_config.json"
                    if variant == "best"
                    else "tuning/tuned_config.json"
                )
                if (directory / config).is_file() and (
                    directory / checkpoint
                ).is_file():
                    cfg = json.loads((directory / config).read_text())
                    return ModelArtifact(
                        cfg.get("modelname") or name, directory.resolve(), name
                    )
    candidates = []
    for root in roots:
        candidates.extend(
            candidate
            for candidate in discover_models(
                root, variant=variant, method=method, names=[name], missing_ok=True
            )
            if candidate.matches(name)
        )
    unique = {candidate.directory: candidate for candidate in candidates}
    if len(unique) != 1:
        if not unique:
            return None
        raise ValueError(
            f"Multiple model artifacts match {name!r}; supply --model-dir or --cv-method"
        )
    return next(iter(unique.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "resolve"))
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--variant", choices=("best", "tuned"), default="best")
    parser.add_argument("--cv-method", choices=CV_METHODS)
    parser.add_argument("--model", action="append", default=[])
    args = parser.parse_args()
    try:
        if args.action == "resolve":
            if len(args.model) != 1:
                parser.error("resolve requires one --model")
            candidate = resolve_model(
                args.model[0],
                roots=args.root,
                variant=args.variant,
                method=args.cv_method,
            )
            if candidate:
                print(candidate.directory)
        else:
            for root in args.root:
                for candidate in discover_models(
                    root, variant=args.variant, method=args.cv_method, names=args.model
                ):
                    print(f"{candidate.name}\t{candidate.directory}")
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
