#!/usr/bin/env python3
"""One-time ONNX export for the Tier 1 encoders, features 11-13 (Appendix A).

    python3 scripts/export_onnx.py            # export everything missing
    python3 scripts/export_onnx.py --check    # preflight only, no downloads

WHY ONNX AND NOT PYTORCH (6.1). The Tier 1 claim is "~0 ms of *waiting*",
achieved by overlapping generation -- but only if a forward pass is 5-15 ms
rather than 50. ONNX on CPU gets there; PyTorch eager does not reliably, and
pulls in a 2 GB image. So torch is an EXPORT-TIME dependency only. It is
deliberately absent from requirements.lock, which pins the serving stack.

WHAT THIS UNBLOCKS. Until these files exist, tier1 reports `unavailable` for
pii, injection and policy; 10.2 fails those invariants CLOSED; and every
request BLOCKs. Demo 3 measured exactly that: 1000/1000 blocked, Tier 2 firing
0%. The system is behaving as specified and is useless in that state.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ONNX_DIR = ROOT / "artifacts" / "onnx"

# Model ids come from Appendix A. Changing one changes `sensor_version`, and
# therefore every configuration fingerprint, and therefore every trust row.
MODELS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2",
               "feature 3/4 retrieval support, and the encoder behind 11 and 13"),
    "prompt_injection": ("protectai/deberta-v3-base-prompt-injection-v2",
                         "feature 12, injection_score"),
    # Appendix A used to say "NER over MiniLM", which cannot be built: MiniLM
    # is a sentence-embedding model and does no token classification. So the
    # NER half of feature 11 never existed, pii_score was `unavailable` on
    # every regex miss, no_pii_egress failed closed on every request, and
    # demo_03 blocked 1000/1000. Corrected in the document 2026-08-27.
    "pii": ("dslim/bert-base-NER",
            "feature 11, the NER half -- names, locations and organisations "
            "that no regex can enumerate"),
}

EXPORT_REQUIREMENTS = ("optimum[onnxruntime]", "transformers", "torch")


def preflight() -> list[str]:
    import importlib.util
    missing = [m for m in ("torch", "transformers", "optimum")
               if importlib.util.find_spec(m) is None]
    return missing


def print_install_help(missing: list[str]) -> None:
    print(f"\n  Export dependencies missing: {', '.join(missing)}")
    print("\n  This machine's Python is externally managed (PEP 668), so pip refuses")
    print("  both a plain install and --user. Two ways forward:\n")
    print("    A. A virtualenv, which needs the venv package once:")
    print("         sudo apt install python3.12-venv")
    print("         python3 -m venv ~/.venv-vouch-export")
    print("         ~/.venv-vouch-export/bin/pip install torch \\")
    print("             --index-url https://download.pytorch.org/whl/cpu")
    print(f"         ~/.venv-vouch-export/bin/pip install {' '.join(EXPORT_REQUIREMENTS[:2])}")
    print("         ~/.venv-vouch-export/bin/python scripts/export_onnx.py\n")
    print("       PUT THE VENV ON THE LINUX FILESYSTEM, not under /mnt/c. A venv on")
    print("       DrvFs installs cleanly and then fails at import with")
    print("       'libtorch_global_deps.so: cannot open shared object file' --")
    print("       DrvFs does not support the symlinks and permissions ELF loading")
    print("       needs. Cost an install cycle to diagnose, 2026-08-25.")
    print("       Use the CPU wheel index: the default PyPI torch is a ~2.5 GB CUDA")
    print("       build, and 6.1 explicitly does not want torch anywhere near serving.\n")
    print("    B. Override the guard, which writes into the system Python:")
    print(f"         pip install --break-system-packages {' '.join(EXPORT_REQUIREMENTS)}\n")
    print("  A is preferable. Everything in this repo currently runs on the system")
    print("  Python, and torch pulls ~2 GB of transitive dependencies into it.")


def already_exported() -> dict[str, bool]:
    return {name: (ONNX_DIR / name / "model.onnx").exists() for name in MODELS}


def export_one(name: str, model_id: str) -> None:
    from optimum.onnxruntime import (
        ORTModelForFeatureExtraction,
        ORTModelForSequenceClassification,
        ORTModelForTokenClassification,
    )
    from transformers import AutoTokenizer

    target = ONNX_DIR / name
    target.mkdir(parents=True, exist_ok=True)
    print(f"  exporting {model_id} -> {target}")

    # The head differs per feature and getting it wrong exports a model whose
    # outputs mean something else entirely: MiniLM is feature extraction,
    # injection is sequence classification, NER is TOKEN classification.
    cls = {
        "minilm": ORTModelForFeatureExtraction,
        "pii": ORTModelForTokenClassification,
    }.get(name, ORTModelForSequenceClassification)
    model = cls.from_pretrained(model_id, export=True)
    model.save_pretrained(target)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(target)

    size_mb = sum(f.stat().st_size for f in target.rglob("*")) / 1e6
    print(f"    done, {size_mb:.0f} MB fp32")
    quantize(name, target)


def quantize(name: str, target: Path) -> None:
    """Dynamic int8. Not an optimisation -- a requirement, measured 2026-08-25.

    fp32 deberta-v3-base runs a forward pass in **54.5 ms** on ONNX CPU. 6.1's
    premise for choosing ONNX is a pass of "5-15 ms, not 50", and the overlap
    measurement in 12.4 showed added latency floors at ONE forward pass. So the
    fp32 injection encoder alone puts added latency an order of magnitude over
    the ~2 ms budget, whatever the rest of the pipeline does.

    It is also the only way the artefact is distributable: fp32 model.onnx is a
    738 MB single file and GitHub rejects anything over 100 MB.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    src = target / "model.onnx"
    dst = target / "model_int8.onnx"
    if dst.exists():
        return
    print(f"    quantizing {name} to int8...")
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    before, after = src.stat().st_size / 1e6, dst.stat().st_size / 1e6
    print(f"    {before:.0f} MB -> {after:.0f} MB  ({before / after:.1f}x smaller)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="preflight only")
    ap.add_argument("--force", action="store_true", help="re-export existing models")
    args = ap.parse_args()

    print("Tier 1 ONNX export (Appendix A features 11-13)\n")
    status = already_exported()
    for name, (model_id, purpose) in MODELS.items():
        mark = "present" if status[name] else "MISSING"
        print(f"  [{mark:7}] {name:17} {model_id}")
        print(f"            {purpose}")

    # Feature 13 is not a downloadable model: 13 is a logistic head fitted on
    # the corpus policy slice, so it cannot exist before the corpus does.
    print(f"\n  [{'pending':7}] policy_score      feature 13 is a logistic head over MiniLM,")
    print("            fitted on the corpus policy slice -- it cannot be exported")
    print("            before generate_corpus.py has run.")

    missing_deps = preflight()
    if missing_deps:
        print_install_help(missing_deps)
        return 1 if not args.check else 0

    if args.check:
        print("\n  Dependencies present. Run without --check to export.")
        return 0

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    for name, (model_id, _purpose) in MODELS.items():
        if status[name] and not args.force:
            continue
        if args.force:
            shutil.rmtree(ONNX_DIR / name, ignore_errors=True)
        export_one(name, model_id)

    print("\n  Re-run scripts/demo_03_latency.py --drive 1000 to re-measure with real")
    print("  encoders. Two independent measurements predict p50 will now exceed the")
    print("  5 ms budget, because added latency floors at one forward pass and 6.1")
    print("  puts ONNX CPU at 5-15 ms. If it does, 12.5 says restate the number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
