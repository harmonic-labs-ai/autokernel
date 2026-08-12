#!/usr/bin/env python3
"""
AutoKernel -- Export optimized kernels to HuggingFace Kernels format.

Takes an optimized AutoKernel Triton kernel and packages it into the
HuggingFace Kernels project structure for publishing to the HuggingFace Hub.

Usage:
    # Export the current kernel.py
    uv run export_hf.py --name my_matmul

    # Export a specific kernel file
    uv run export_hf.py --name my_matmul --kernel workspace/kernel_matmul_1.py

    # Export with a specific repo ID for upload instructions
    uv run export_hf.py --name my_matmul --repo-id rightnow-ai/matmul-kernel

    # Custom output directory
    uv run export_hf.py --name my_matmul --output workspace/hf_export/

    # After export, upload to HuggingFace Hub:
    #   cd workspace/hf_export/my_matmul
    #   huggingface-cli upload rightnow-ai/matmul-kernel . .

HuggingFace Kernels: https://huggingface.co/docs/kernels/en/index
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import textwrap
from typing import Callable, Dict, Optional


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_KERNEL_PATH = os.path.join(SCRIPT_DIR, "kernel.py")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "workspace", "hf_export")

DEFAULT_BACKEND = "triton"


# ---------------------------------------------------------------------------
# Backend / kernel-type detection
# ---------------------------------------------------------------------------

def detect_backend(source: str) -> str:
    """Return the backend a kernel file declares, defaulting to Triton.

    Kernels carry an explicit ``BACKEND = "..."`` line only when they are not
    the default backend, so an undeclared kernel is a Triton kernel.
    """
    match = re.search(r'^BACKEND\s*=\s*["\'](\w+)["\']', source, re.MULTILINE)
    if match:
        return match.group(1).lower()
    return DEFAULT_BACKEND


def detect_kernel_type(source: str) -> Optional[str]:
    """Extract the KERNEL_TYPE from the source file, if declared."""
    match = re.search(r'^KERNEL_TYPE\s*=\s*["\'](\w+)["\']', source, re.MULTILINE)
    if match:
        return match.group(1)
    return None


# ---------------------------------------------------------------------------
# Triton source extraction
# ---------------------------------------------------------------------------

def extract_triton_code(source: str) -> str:
    """
    Extract the Triton kernel code from a Python file.

    Returns everything from the first import statement onward, skipping
    the module docstring and KERNEL_TYPE/BACKEND declarations.
    """
    lines = source.split("\n")

    # Find the first import line
    import_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            import_idx = i
            break

    if import_idx is not None:
        return "\n".join(lines[import_idx:])

    # Fallback: return everything after KERNEL_TYPE line
    for i, line in enumerate(lines):
        if line.strip().startswith("KERNEL_TYPE"):
            return "\n".join(lines[i + 1 :])

    return source


# ---------------------------------------------------------------------------
# File generation: __init__.py
# ---------------------------------------------------------------------------

def generate_init_py(name: str, repo_id: str) -> str:
    """Generate the Python __init__.py for the HF Kernels module."""
    return textwrap.dedent(f'''\
        """
        {name} - Optimized Triton GPU kernel exported from AutoKernel
        https://github.com/RightNow-AI/autokernel

        Usage:
            from kernels import get_kernel
            module = get_kernel("{repo_id}")
            result = module.kernel_fn(input)
        """
        from .kernel import kernel_fn  # noqa: F401
    ''')


# ---------------------------------------------------------------------------
# Export pipeline: Triton
# ---------------------------------------------------------------------------

def _export_triton_kernel(
    source: str,
    name: str,
    output_dir: str,
    repo_id: str,
) -> None:
    """
    Export a Triton kernel to HF Kernels format.

    Triton kernels are already Python, so the export is simple: package the
    Triton code as a Python module. No ahead-of-time compilation is needed --
    the Triton JIT handles everything at runtime.
    """
    # Create directory structure
    project_dir = os.path.join(output_dir, name)
    module_dir = os.path.join(project_dir, name)

    os.makedirs(module_dir, exist_ok=True)

    # 1. Write the Triton kernel as kernel.py inside the module
    triton_code = extract_triton_code(source)
    kernel_py_path = os.path.join(module_dir, "kernel.py")
    with open(kernel_py_path, "w", encoding="utf-8") as f:
        f.write(triton_code.strip())
        f.write("\n")
    print(f"  Created {os.path.relpath(kernel_py_path, output_dir)}")

    # 2. Write __init__.py
    init_py_path = os.path.join(module_dir, "__init__.py")
    with open(init_py_path, "w", encoding="utf-8") as f:
        f.write(generate_init_py(name, repo_id))
    print(f"  Created {os.path.relpath(init_py_path, output_dir)}")

    # 3. Write a minimal pyproject.toml for the Triton package
    pyproject_path = os.path.join(project_dir, "pyproject.toml")
    pyproject_content = textwrap.dedent(f"""\
        [project]
        name = "{name}"
        version = "0.1.0"
        description = "Optimized Triton GPU kernel exported from AutoKernel"
        requires-python = ">=3.10"
        dependencies = [
            "torch>=2.4.0",
            "triton>=3.3.0",
        ]
    """)
    with open(pyproject_path, "w", encoding="utf-8") as f:
        f.write(pyproject_content)
    print(f"  Created {os.path.relpath(pyproject_path, output_dir)}")

    # 4. Write a README for the Hub repo
    readme_path = os.path.join(project_dir, "README.md")
    readme_content = textwrap.dedent(f"""\
        # {name}

        Optimized Triton GPU kernel exported from [AutoKernel](https://github.com/RightNow-AI/autokernel).

        ## Usage

        ```python
        from {name}.kernel import kernel_fn

        result = kernel_fn(input_tensor)
        ```

        ## Requirements

        - PyTorch >= 2.4.0
        - Triton >= 3.3.0
        - NVIDIA GPU
    """)
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"  Created {os.path.relpath(readme_path, output_dir)}")

    print()
    print("  Exported Triton kernel as a Python package.")
    print("  Entry point: kernel_fn()")


def _print_triton_next_steps(project_dir: str, name: str, repo_id: str) -> None:
    print("  1. Review the exported files:")
    print(f"     ls {project_dir}/")
    print()
    print("  2. Upload to HuggingFace Hub:")
    print("     # First: pip install huggingface-hub && huggingface-cli login")
    print(f"     cd {project_dir}")
    print(f"     huggingface-cli upload {repo_id} . .")
    print()
    print("  3. Use from anywhere:")
    print(f"     from {name}.kernel import kernel_fn")
    print("     result = kernel_fn(input)")


# Backend -> (exporter, next-steps printer). Adding a backend means adding its
# exporter here; export_kernel dispatches on the kernel's declared BACKEND.
EXPORTERS: Dict[str, Callable[[str, str, str, str], None]] = {
    "triton": _export_triton_kernel,
}
NEXT_STEPS: Dict[str, Callable[[str, str, str], None]] = {
    "triton": _print_triton_next_steps,
}


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export_kernel(
    kernel_path: str,
    name: str,
    output_dir: str,
    repo_id: Optional[str] = None,
) -> str:
    """
    Main export pipeline.

    Parameters
    ----------
    kernel_path : str
        Path to the AutoKernel kernel file (kernel.py or similar).
    name : str
        Name for the exported kernel project (used as the module name).
        Must be a valid Python identifier.
    output_dir : str
        Directory where the HF Kernels project will be created.
    repo_id : str, optional
        HuggingFace repo ID (e.g., "rightnow-ai/matmul-kernel").
        Used in documentation and __init__.py usage examples.

    Returns
    -------
    str
        Path to the exported project directory.
    """
    # Validate name is a valid Python identifier
    if not name.isidentifier():
        print(f"ERROR: '{name}' is not a valid Python identifier.")
        print("       Use a name like 'my_matmul' or 'fused_attention'.")
        sys.exit(1)

    # Default repo_id
    if repo_id is None:
        repo_id = f"your-username/{name}"

    # Read the kernel file
    if not os.path.exists(kernel_path):
        print(f"ERROR: Kernel file not found: {kernel_path}")
        sys.exit(1)

    with open(kernel_path, "r", encoding="utf-8") as f:
        source = f.read()

    if not source.strip():
        print(f"ERROR: Kernel file is empty: {kernel_path}")
        sys.exit(1)

    backend = detect_backend(source)
    if backend not in EXPORTERS:
        print(f"ERROR: Kernel declares BACKEND = '{backend}', which this build "
              f"cannot export.")
        print(f"       Supported backends: {', '.join(sorted(EXPORTERS))}")
        sys.exit(1)

    kernel_type = detect_kernel_type(source)

    print(f"=== AutoKernel HuggingFace Kernels Export ===")
    print()
    print(f"  Kernel file:  {kernel_path}")
    print(f"  Backend:      {backend}")
    if kernel_type:
        print(f"  Kernel type:  {kernel_type}")
    print(f"  Project name: {name}")
    print(f"  Repo ID:      {repo_id}")
    print(f"  Output:       {output_dir}")
    print()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    project_dir = os.path.join(output_dir, name)

    # Check if project directory already exists
    if os.path.exists(project_dir):
        print(f"WARNING: Output directory already exists: {project_dir}")
        print("         Files will be overwritten.")
        print()

    EXPORTERS[backend](source, name, output_dir, repo_id)

    print()
    print("=" * 60)
    print("  Export complete!")
    print("=" * 60)
    print()
    print("  Next steps:")
    print()
    NEXT_STEPS[backend](project_dir, name, repo_id)

    print()
    return project_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export an optimized AutoKernel Triton kernel to HuggingFace "
            "Kernels format."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              # Export the default kernel.py
              uv run export_hf.py --name my_matmul

              # Export a specific kernel file with repo ID
              uv run export_hf.py --name my_matmul --kernel workspace/kernel_matmul_1.py \\
                                  --repo-id rightnow-ai/matmul-kernel

              # Custom output directory
              uv run export_hf.py --name my_matmul --output /tmp/hf_export/
        """),
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help=(
            "Name for the exported kernel project. Must be a valid Python identifier "
            "(e.g., 'my_matmul', 'fused_attention')."
        ),
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default=DEFAULT_KERNEL_PATH,
        help=f"Path to the kernel file to export (default: kernel.py)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for the HF Kernels project (default: workspace/hf_export/)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help=(
            "HuggingFace repo ID (e.g., 'rightnow-ai/matmul-kernel'). "
            "Used in documentation and usage examples."
        ),
    )

    args = parser.parse_args()

    export_kernel(
        kernel_path=args.kernel,
        name=args.name,
        output_dir=args.output,
        repo_id=args.repo_id,
    )


if __name__ == "__main__":
    main()
