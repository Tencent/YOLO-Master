"""Command-line entry points for the P2 evidence run and local demo."""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO-Master E3 P2 tooling")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="generate a fresh P2 evidence bundle")
    run_parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "p2" / "p2.yaml")
    run_parser.add_argument("--run-id")
    run_parser.add_argument("--no-latest", action="store_true")
    robustness_parser = subparsers.add_parser(
        "robustness", help="run CPU resolution and horizontal-flip diagnostics"
    )
    robustness_parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "p2" / "robustness.yaml"
    )
    robustness_parser.add_argument("--run-id")
    robustness_parser.add_argument("--no-latest", action="store_true")
    appearance_parser = subparsers.add_parser(
        "appearance", help="run CPU brightness, contrast and blur diagnostics"
    )
    appearance_parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "p2" / "appearance.yaml"
    )
    appearance_parser.add_argument("--run-id")
    appearance_parser.add_argument("--no-latest", action="store_true")
    drilldown_parser = subparsers.add_parser(
        "layer-drilldown", help="analyze layer attribution and margin-conditioned switch locations"
    )
    drilldown_parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "p2" / "layer_drilldown.yaml"
    )
    drilldown_parser.add_argument("--run-id")
    drilldown_parser.add_argument("--no-latest", action="store_true")
    scale_parser = subparsers.add_parser(
        "image-scale", help="run the CPU coco128 image-level MoA appearance audit"
    )
    scale_parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "p2" / "image_scale.yaml"
    )
    scale_parser.add_argument("--run-id")
    scale_parser.add_argument("--no-latest", action="store_true")
    driver_parser = subparsers.add_parser(
        "image-driver", help="analyze within-transform image-level input and routing change associations"
    )
    driver_parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "p2" / "image_driver.yaml"
    )
    driver_parser.add_argument("--run-id")
    driver_parser.add_argument("--no-latest", action="store_true")
    dose_parser = subparsers.add_parser(
        "dose-response", help="run the predeclared 32-image MoA appearance-strength ladder"
    )
    dose_parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "p2" / "dose_response.yaml"
    )
    dose_parser.add_argument("--run-id")
    dose_parser.add_argument("--no-latest", action="store_true")
    coupling_parser = subparsers.add_parser(
        "output-coupling", help="run image-level router-to-detector-output association diagnostics"
    )
    coupling_parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "p2" / "output_coupling.yaml"
    )
    coupling_parser.add_argument("--run-id")
    coupling_parser.add_argument("--no-latest", action="store_true")
    holdout_parser = subparsers.add_parser(
        "dose-holdout", help="validate the held middle dose from low/high endpoint evidence"
    )
    holdout_parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "p2" / "dose_holdout.yaml"
    )
    holdout_parser.add_argument("--run-id")
    holdout_parser.add_argument("--no-latest", action="store_true")
    residual_parser = subparsers.add_parser(
        "blur-residual", help="diagnose held-middle Gaussian-blur residuals from locked input-only features"
    )
    residual_parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "p2" / "blur_residual.yaml"
    )
    residual_parser.add_argument("--run-id")
    residual_parser.add_argument("--no-latest", action="store_true")
    demo_parser = subparsers.add_parser("demo", help="serve the latest evidence demo")
    demo_parser.add_argument("--host", default="127.0.0.1")
    demo_parser.add_argument("--port", type=int, default=8766)
    demo_parser.add_argument("--run-dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "run":
        from .runner import run

        output = run(args.config, run_id=args.run_id, update_latest=not args.no_latest)
        print(f"P2 evidence: {output}")
    elif args.command == "robustness":
        from .robustness_runner import run as run_robustness

        output = run_robustness(args.config, run_id=args.run_id, update_latest=not args.no_latest)
        print(f"P2 robustness evidence: {output}")
    elif args.command == "appearance":
        from .appearance_runner import run as run_appearance

        output = run_appearance(args.config, run_id=args.run_id, update_latest=not args.no_latest)
        print(f"P2 appearance evidence: {output}")
    elif args.command == "layer-drilldown":
        from .layer_drilldown import run as run_layer_drilldown

        output = run_layer_drilldown(args.config, run_id=args.run_id, update_latest=not args.no_latest)
        print(f"P2 layer attribution evidence: {output}")
    elif args.command == "image-scale":
        from .scale_runner import run as run_image_scale

        output = run_image_scale(args.config, run_id=args.run_id, update_latest=not args.no_latest)
        print(f"P2 image-scale evidence: {output}")
    elif args.command == "image-driver":
        from .image_driver_analysis import run as run_image_driver

        output = run_image_driver(args.config, run_id=args.run_id, update_latest=not args.no_latest)
        print(f"P2 image-driver evidence: {output}")
    elif args.command == "dose-response":
        from .dose_response_runner import run as run_dose_response

        output = run_dose_response(args.config, run_id=args.run_id, update_latest=not args.no_latest)
        print(f"P2 dose-response evidence: {output}")
    elif args.command == "output-coupling":
        from .output_coupling_runner import run as run_output_coupling

        output = run_output_coupling(args.config, run_id=args.run_id, update_latest=not args.no_latest)
        print(f"P2 output-coupling evidence: {output}")
    elif args.command == "dose-holdout":
        from .dose_holdout_runner import run as run_dose_holdout

        output = run_dose_holdout(args.config, run_id=args.run_id, update_latest=not args.no_latest)
        print(f"P2 dose-holdout evidence: {output}")
    elif args.command == "blur-residual":
        from .blur_residual_runner import run as run_blur_residual

        output = run_blur_residual(args.config, run_id=args.run_id, update_latest=not args.no_latest)
        print(f"P2 blur-residual evidence: {output}")
    else:
        from .demo import serve

        serve(host=args.host, port=args.port, run_dir=args.run_dir)


if __name__ == "__main__":
    main()
