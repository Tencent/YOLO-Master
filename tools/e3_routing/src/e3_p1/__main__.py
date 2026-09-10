"""Command-line entry point for the E3 P1 benchmark."""

from .benchmark import parse_args, run

args = parse_args()
run(args.config.resolve(), run_id=args.run_id, update_latest=not args.no_update_latest)
