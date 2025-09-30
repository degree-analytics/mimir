#!/usr/bin/env python3
"""Build Slack message payload for release/CI notifications.

This script generates a Slack message payload JSON file for posting notifications
about CI runs and releases. It formats the message with appropriate emoji, version
information, commit details, and links.

Designed to be called from GitHub Actions workflows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Slack message payload")
    parser.add_argument("--status", required=True, help="Workflow status (success, failure, cancelled)")
    parser.add_argument("--channel", required=True, help="Slack channel ID")
    parser.add_argument("--environment", required=True, help="Environment name (e.g., Main)")
    parser.add_argument("--is-release", required=True, help="Whether this is a release (true/false)")
    parser.add_argument("--version", required=True, help="Version number")
    parser.add_argument("--subject", required=True, help="Commit subject line")
    parser.add_argument("--short-sha", required=True, help="Short commit SHA")
    parser.add_argument("--head-sha", required=True, help="Full commit SHA")
    parser.add_argument("--repo", required=True, help="Repository name (owner/repo)")
    parser.add_argument("--actor", required=True, help="GitHub username")
    parser.add_argument("--run-url", required=True, help="Workflow run URL")
    parser.add_argument("--output", default="payload.json", help="Output file path")
    return parser.parse_args()


def build_payload(args: argparse.Namespace) -> Dict[str, Any]:
    """Build the Slack message payload."""

    status = args.status.lower()
    emoji = ":large_green_circle:" if status == "success" else ":red_circle:"

    # Check for version release
    is_release = args.is_release.lower() == "true"
    version = args.version

    # Build title with version info
    environment = args.environment
    if is_release and version:
        text = f"{emoji} {environment} Released v{version}"
    else:
        text = f"{emoji} {environment} CI {status.upper()}"

    # Build commit URL
    commit_url = None
    if args.repo and args.head_sha:
        commit_url = f"https://github.com/{args.repo}/commit/{args.head_sha}"

    header_lines = []

    # Add version info if this is a release
    if is_release and version:
        header_lines.append(f"• *Version*: v{version}")

    # Add commit info
    if args.short_sha:
        if commit_url:
            header_lines.append(f"• *Commit*: <{commit_url}|{args.short_sha}> – \"{args.subject}\"")
        elif args.run_url:
            header_lines.append(f"• *Commit*: <{args.run_url}|{args.short_sha}> – \"{args.subject}\"")
        elif args.subject:
            header_lines.append(f"• *Commit*: {args.short_sha} – \"{args.subject}\"")
    elif args.subject:
        header_lines.append(f"• *Commit*: {args.short_sha or 'unknown'} – \"{args.subject}\"")

    if args.actor:
        header_lines.append(f"• *Actor*: @{args.actor}")
    if args.repo:
        header_lines.append(f"• *Repo*: {args.repo}")

    header_text = "\n".join(header_lines)

    payload = {
        "channel": args.channel,
        "text": text,
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"{emoji} *{text.replace(emoji + ' ', '')}*\n{header_text}"
                }
            }
        ]
    }

    if args.run_url:
        payload["blocks"].append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"<{args.run_url}|View run details>"
                }
            ]
        })

    return payload


def main() -> int:
    """Main entry point."""
    args = parse_args()

    try:
        payload = build_payload(args)

        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        return 0
    except Exception as e:
        print(f"Error building Slack payload: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
