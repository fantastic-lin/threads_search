import argparse
import yaml
import os
import sys

# Allow Python to import from the app package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import core functions implemented elsewhere
from app.threads_app import run_keyword_export, run_user_posts


def add_common_io_args(p: argparse.ArgumentParser) -> None:
    """Add common I/O arguments shared by both subcommands."""
    p.add_argument("-c", "--config", default="config.yml", help="Path to configuration file")
    p.add_argument(
        "--out",
        default=None,
        choices=["csv", "xlsx", "json", "gsheets", "preview"],
        help="Export format; if not specified, read from config file",
    )
    p.add_argument("--export_dir", default=None, help="Export directory (override config file)")
    p.add_argument("--basename", default=None, help="Base name of export file (without extension)")
    p.add_argument("--clean", action="store_true", help="Clean text (strip HTML / handle emojis, etc.)")
    p.add_argument("--no-clean", dest="clean", action="store_false", help="Disable text cleaning")
    p.set_defaults(clean=True)
    p.add_argument("--time_format", default=None, choices=["local", "utc"], help="Time format (local|utc)")
    p.add_argument("--emoji_format", default=None, choices=["emoji", "alias"], help="Emoji format (emoji|alias)")


def load_cfg(path: str) -> dict:
    """Load YAML configuration safely; return an empty dict on error/missing."""
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        print(f"[WARN] Failed to load config: {e}")
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Threads Search Tool: supports both keyword search and username post fetching"
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # keyword subcommand
    pk = subparsers.add_parser("keyword", help="Search by keyword (run_keyword_export)")
    pk.add_argument("keywords", nargs="+", help="Query keywords (one or more)")
    add_common_io_args(pk)

    # Keyword-specific arguments
    pk.add_argument(
        "--max_posts",
        type=int,
        default=None,
        help="Fetch limit (total or per keyword, depending on limit_mode)",
    )
    pk.add_argument(
        "--limit_mode",
        default=None,
        choices=["per", "total"],
        help="per = per keyword, total = overall",
    )
    pk.add_argument("--search_type", default=None, choices=["TOP", "LATEST"], help="Search type")
    pk.add_argument("--search_mode", default=None, choices=["KEYWORD"], help="Search mode (reserved)")
    pk.add_argument(
        "--media_type",
        default=None,
        choices=["TEXT", "IMAGE", "VIDEO", "ALL"],
        help="Media type filter",
    )
    pk.add_argument("--since", default=None, help="Start date YYYY-MM-DD")
    pk.add_argument("--until", default=None, help="End date YYYY-MM-DD")

    # Term frequency (keyword mode)
    pk.add_argument("--termfreq_mode", action="store_true", help="Enable simple term frequency statistics")
    pk.add_argument(
        "--termfreq_scope",
        default=None,
        choices=["new", "history"],
        help="Scope for term frequency statistics",
    )
    pk.add_argument("--termfreq_topn", type=int, default=None, help="Top-N for term frequency")
    pk.add_argument("--termfreq_outdir", default=None, help="Output directory for term frequency results")
    pk.add_argument("--send_email", action="store_true", help="Send the results via email")
    
    # username subcommand
    pu = subparsers.add_parser("username", help="Fetch posts by username (run_user_posts)")
    pu.add_argument("handles", nargs="+", help="Usernames (one or more), with or without @ prefix")
    add_common_io_args(pu)

    # Username-specific arguments
    pu.add_argument("--max_posts", type=int, default=None, help="Total fetch limit (across all users)")
    pu.add_argument("--per_max", type=int, default=None, help="Maximum number of posts per user")
    pu.add_argument("--sleep_s", type=float, default=None, help="Sleep seconds between each user")
    pu.add_argument("--user_store_dir", default=None, help="User history cache directory (for deduplication)")
    pu.add_argument("--persist_ext", default=None, help="Persistence file extension (json, etc.)")

    # Term frequency (username mode)
    pu.add_argument("--termfreq_mode", action="store_true", help="Enable simple term frequency statistics")
    pu.add_argument(
        "--termfreq_scope",
        default=None,
        choices=["new", "history"],
        help="Scope for term frequency statistics",
    )
    pu.add_argument("--termfreq_topn", type=int, default=None, help="Top-N for term frequency")
    pu.add_argument("--termfreq_outdir", default=None, help="Output directory for term frequency results")
    pu.add_argument("--send_email", action="store_true", help="Send the results via email")

    args = parser.parse_args()

    # Load config (shared)
    cfg = load_cfg(getattr(args, "config", None))
    p = (cfg.get("params") or {})

    # Common export parameters (CLI overrides YAML)
    out = (args.out or p.get("out") or "csv").lower()
    keyword_export_dir = args.export_dir or p.get("keywordExportDir", "data/keyword_store")
    username_export_dir = args.export_dir or p.get("usernameExportDir", "data/username_store")
    basename = args.basename or p.get("basename") or None
    clean = bool(args.clean if args.clean is not None else str(p.get("clean", "1")) == "1")
    time_format = (args.time_format or p.get("time_format") or "local").lower()
    emoji_format = (args.emoji_format or p.get("emoji_format") or "emoji").lower()

    send_email_to = None
    if args.send_email:
        send_email_to = input("Please enter the email address to send results to: ").strip()

    # Keyword mode
    if args.cmd == "keyword":
        res = run_keyword_export(
            keywords=args.keywords,
            max_posts=int(args.max_posts if args.max_posts is not None else p.get("max_posts", 25)),
            limit_mode=(args.limit_mode or p.get("limitMode", "per")).lower(),
            search_type=(args.search_type or p.get("searchType", "TOP")).upper(),
            search_mode=(args.search_mode or p.get("searchMode", "KEYWORD")).upper(),
            media_type=(args.media_type or p.get("mediaType", "TEXT")).upper(),
            since=(args.since if args.since is not None else p.get("since")),
            until=(args.until if args.until is not None else p.get("until")),
            out=out,
            clean=clean,
            time_format=time_format,
            emoji_format=emoji_format,
            export_dir=keyword_export_dir,
            output_basename=basename,
            termfreq_mode=bool(args.termfreq_mode),
            termfreq_scope=(args.termfreq_scope or p.get("scope", "new")).lower(),
            termfreq_topn=int(args.termfreq_topn if args.termfreq_topn is not None else p.get("topn", 50)),
            termfreq_outdir=(args.termfreq_outdir or p.get("outdir", keyword_export_dir)),
            send_email=bool(args.send_email),
            send_email_to=send_email_to
        )

        # Output summary
        if res.get("file"):
            print(f"Saved file: {res['file']}")
        elif res.get("sheet"):
            print(f"GSheets URL/ID: {res['sheet']}")
        elif res.get("preview"):
            print("Preview:", res["preview"])

        # Term frequency info
        if args.termfreq_mode:
            print(
                f"[termfreq] scope={res.get('termfreq_scope')}, "
                f"topn={res.get('termfreq_topn')}"
            )
            if "term_freq_file_name" in res:
                print(
                    f"[termfreq] output: "
                    f"{os.path.join(res.get('termfreq_outdir', keyword_export_dir), res['term_freq_file_name'])}"
                )

    # Username mode
    elif args.cmd == "username":
        max_posts = int(args.max_posts if args.max_posts is not None else p.get("max_posts", 5))
        per_max = int(args.per_max if args.per_max is not None else p.get("per_max", 2))
        sleep_s = float(args.sleep_s if args.sleep_s is not None else p.get("sleep_s", 0.15))
        user_store_dir = args.user_store_dir or p.get("user_store_dir", "data/user_store")
        persist_ext = args.persist_ext or p.get("persist_ext", "json")

        res = run_user_posts(
            handles=args.handles,
            max_posts=max_posts,
            per_max=per_max,
            out=out,
            clean=clean,
            time_format=time_format,
            emoji_format=emoji_format,
            sleep_s=sleep_s,
            user_store_dir=user_store_dir,
            persist_ext=persist_ext,
            export_dir=username_export_dir,
            output_basename=basename,
            termfreq_mode=bool(args.termfreq_mode),
            termfreq_scope=(args.termfreq_scope or p.get("scope", "history")).lower(),
            termfreq_topn=int(args.termfreq_topn if args.termfreq_topn is not None else p.get("topn", 50)),
            termfreq_outdir=(args.termfreq_outdir or username_export_dir),
            send_email=bool(args.send_email),
            send_email_to=send_email_to
        )

        # Output summary
        if res.get("file"):
            print(f"Saved file: {res['file']}")
        elif res.get("sheet"):
            print(f"GSheets URL/ID: {res['sheet']}")
        elif res.get("preview"):
            print("Preview:", res["preview"])

        # Term frequency info
        if args.termfreq_mode:
            print(
                f"[termfreq] scope={res.get('termfreq_scope')}, "
                f"topn={res.get('termfreq_topn')}"
            )
            if "term_freq_file_name" in res:
                print(
                    f"[termfreq] output: "
                    f"{os.path.join(res.get('termfreq_outdir', username_export_dir), res['term_freq_file_name'])}"
                )

    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
