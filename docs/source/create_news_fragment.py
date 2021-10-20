from enum import Enum
from typing import List
from pathlib import Path
import logging
import argparse
import requests

_root = Path(__file__).parent / "release" / "changelog"
_valid_types = ("feature", "bugfix", "deprecation", "misc", "doc", "ignore-towncrier")


class State(str, Enum):
    DESCRIPTION_START = "## Description"
    DESCRIPTION_END = "##"
    COMMENT = "<!--"


def _parse_description(comment: str) -> str:
    def generate(lines: List[str]) -> str:
        if not len(lines):
            raise ValueError("No description lines have been detected.")
        lines = [line.strip() for line in lines]
        if all(not len(line) for line in lines):
            raise ValueError("All description lines are empty after removing leading/trailing whitespace.")

        return "\n".join(lines).strip()

    inside_description = False
    desc_lines: List[str] = []

    for line in comment.split("\n"):
        if line.startswith(State.COMMENT):
            continue
        if inside_description:
            if line.startswith(State.DESCRIPTION_END):
                return generate(desc_lines)
            desc_lines.append(line)
        if line.startswith(State.DESCRIPTION_START):
            inside_description = True

    if inside_description:
        return generate(desc_lines)

    raise ValueError("Unable to parse description from the issue comment.")


def create_news_fragment(issue_number: str, use_title: bool = False, sep: str = "~") -> None:
    try:
        url = f"https://api.github.com/repos/theislab/squidpy/pulls/{issue_number}"
        resp = requests.get(url)
        resp.raise_for_status()

        data = resp.json()
        types = data["labels"]

        if "ignore-towncrier" in [t["name"] for t in types]:
            logging.info(f"Ignoring news fragment generation for issue `{issue_number}`")
            return

        typ = str(types[0]["name"] if len(types) else "bugfix").strip()
        if typ not in _valid_types:
            raise ValueError(f"Expected label to be on of `{_valid_types}`, found `{typ!r}`.")
        logging.info(f"Generating `{typ}` news fragment generation for issue `{issue_number}`")

        title = str(data["title"]).strip()
        description = _parse_description(data["body"])

        fragment = f"{title}\n{len(title) * sep}\n{description}" if use_title else description
        fpath = _root / f"{issue_number}.{typ}.rst"

        logging.info(f"Saving news fragment into `{fpath}`")
        with open(fpath, "w") as fout:
            print(fragment, file=fout)
    except Exception as e:
        logging.error(f"Unable to generate news fragment. Reason: `{e}`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatically create news fragment from an issue number.")
    parser.add_argument(
        "issue_number",
        type=str,
        metavar="ISSUE_NUMBER",
        help="Issue from which to create the news fragment.",
    )
    parser.add_argument(
        "--title",
        action="store_true",
        help="Whether to include the issue title in the fragment.",
    )
    parser.add_argument("-v", "--verbose", action="count", help="Be verbose.")

    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    create_news_fragment(issue_number=args.issue_number, use_title=args.title)
