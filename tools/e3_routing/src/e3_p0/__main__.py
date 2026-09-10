"""Command-line entry point."""

from .runner import parse_args, run

args = parse_args()
run(args.config.resolve(), run_id=args.run_id, update_latest=not args.no_update_latest)
