#!/usr/bin/env python3
"""Discover public Streamlit projects and regenerate the GitHub Pages index.

Human-curated fields live in _data/streamlit_apps.json.  When the script finds
an unknown *live* Streamlit deployment, it adds it with section "unclassified"
rather than guessing a subject classification.  Editors then assign its section,
title, and description in the catalog.  No third-party Python packages are used.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "_data" / "streamlit_apps.json"
PAGE_PATH = ROOT / "streamlit-apps.md"
API_ROOT = "https://api.github.com"
STREAMLIT_URL = re.compile(r"https?://[A-Za-z0-9.-]+\.streamlit\.app(?:/[^\s\]})>'\"`]*)?", re.IGNORECASE)


def request(url: str, token: str | None, timeout: int = 25) -> tuple[int, bytes]:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "mboudour-streamlit-index"}
    # Never forward the repository token to a Streamlit deployment or any other
    # third-party URL. Authentication is needed only for GitHub API calls.
    if token and url.startswith(API_ROOT):
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urlopen(Request(url, headers=headers), timeout=timeout) as response:
            return response.status, response.read()
    except HTTPError as exc:
        return exc.code, exc.read()
    except URLError:
        return 0, b""


def get_json(url: str, token: str | None) -> Any:
    status, body = request(url, token)
    if status != 200:
        raise RuntimeError(f"GitHub API request failed ({status}): {url}")
    return json.loads(body.decode("utf-8"))


def list_repositories(owner: str, token: str | None) -> list[dict[str, Any]]:
    repositories: list[dict[str, Any]] = []
    page = 1
    while True:
        url = f"{API_ROOT}/users/{quote(owner)}/repos?type=owner&per_page=100&page={page}&sort=full_name"
        batch = get_json(url, token)
        if not batch:
            return repositories
        repositories.extend(batch)
        page += 1


def repository_tree(owner: str, repository: dict[str, Any], token: str | None) -> list[str]:
    branch = repository.get("default_branch")
    if not branch:
        return []
    url = f"{API_ROOT}/repos/{quote(owner)}/{quote(repository['name'])}/git/trees/{quote(branch, safe='')}?recursive=1"
    payload = get_json(url, token)
    return [entry["path"] for entry in payload.get("tree", []) if entry.get("type") == "blob"]


def relevant_paths(paths: list[str]) -> list[str]:
    selected: dict[int, list[str]] = {1: [], 2: [], 3: [], 4: []}
    for path in paths:
        lower = path.lower()
        name = lower.rsplit("/", 1)[-1]
        priority: int | None = None
        if name.startswith("readme"):
            priority = 1
        elif "streamlit" in lower:
            priority = 2
        elif name.startswith("requirements") or name in {"pyproject.toml", "environment.yml", "environment.yaml", "setup.py", "setup.cfg"}:
            priority = 3
        elif lower.endswith(".py") and any(key in name for key in ("app", "main", "streamlit")):
            priority = 4
        if priority is not None:
            selected[priority].append(path)
    # A small, balanced sample is enough to identify a Streamlit project and find
    # deployment links, while keeping the scheduled scan fast and rate-limit-safe.
    limits = {1: 3, 2: 6, 3: 4, 4: 4}
    return [path for priority in (1, 2, 3, 4) for path in sorted(selected[priority])[:limits[priority]]]


def has_streamlit_implementation(sources: list[tuple[str, str]]) -> bool:
    """Require code or dependency evidence; a page that merely links to an app is not one."""
    for path, text in sources:
        lower_path = path.lower()
        lower_text = text.lower()
        if lower_path.endswith(".py") and re.search(r"\b(?:import|from)\s+streamlit\b", lower_text):
            return True
        if lower_path.rsplit("/", 1)[-1].startswith("requirements") and re.search(r"\bstreamlit\b", lower_text):
            return True
        if lower_path.rsplit("/", 1)[-1] in {"pyproject.toml", "environment.yml", "environment.yaml", "setup.py", "setup.cfg"} and re.search(r"\bstreamlit\b", lower_text):
            return True
        if lower_path.endswith(".py") and "streamlit run" in lower_text:
            return True
    return False


def raw_file(owner: str, repository: dict[str, Any], path: str, token: str | None) -> str:
    branch = repository["default_branch"]
    url = "https://raw.githubusercontent.com/{}/{}/{}/{}".format(
        quote(owner), quote(repository["name"]), quote(branch, safe=""), quote(path, safe="/")
    )
    status, body = request(url, token)
    if status != 200:
        return ""
    return body.decode("utf-8", errors="replace")[:1_000_000]


def normalized_urls(text: str) -> list[str]:
    urls: list[str] = []
    for url in STREAMLIT_URL.findall(text):
        normalized = url.rstrip(".,;:!?)]}\"'")
        if not normalized.endswith("/"):
            normalized += "/"
        if normalized not in urls:
            urls.append(normalized)
    return urls


def guessed_url(repository_name: str) -> str:
    slug = re.sub(r"[^a-z0-9-]", "", repository_name.lower().replace("_", "-"))
    return f"https://{slug}.streamlit.app/"


def is_live(url: str, token: str | None) -> bool:
    status, _ = request(url, token, timeout=30)
    return 200 <= status < 400


def readable_title(repository_name: str) -> str:
    return re.sub(r"[-_]+", " ", repository_name).title()


def source_for(repository_name: str, repository_url: str) -> str:
    return f"[View the GitHub project]({repository_url})"


def render(catalog: dict[str, Any], active: list[dict[str, Any]]) -> str:
    sections = catalog["sections"]
    section_ids = {section["id"] for section in sections}
    if "unclassified" not in section_ids:
        raise RuntimeError("The catalog must contain an 'unclassified' section.")

    lines = [
        "---",
        "layout: page",
        "title: Moses Boudourides — Public Streamlit Applications",
        "permalink: /streamlit-apps/",
        "---",
        "",
        "<!-- This file is generated by scripts/update_streamlit_index.py. Edit _data/streamlit_apps.json to curate titles, descriptions, and sections. -->",
        "",
        "This page collects Moses Boudourides's public Streamlit applications linked to his public GitHub projects.",
        "",
        "> **Automatic discovery with curated classification.** The index is refreshed weekly from the public, non-fork repositories in the [mboudour GitHub account](https://github.com/mboudour). Existing applications retain their assigned section. A newly discovered live application is shown under **New and unclassified applications** until its subject classification, title, and description are reviewed in the catalog.",
        "",
    ]
    for section in sections:
        entries = [entry for entry in active if entry["section"] == section["id"]]
        if not entries:
            continue
        lines.extend([f"## {section['title']}", ""])
        if section["id"] == "unclassified":
            lines.extend([
                "These live deployments were found automatically. They are deliberately not assigned to a research area until reviewed.",
                "",
            ])
        for entry in sorted(entries, key=lambda item: item["title"].casefold()):
            lines.extend([
                f"### [{entry['title']}]({entry['app_url']})",
                entry["description"],
                "",
                f"[Open the app]({entry['app_url']}) · {source_for(entry['repository'], entry['repository_url'])}",
                "",
            ])
    lines.extend([
        "---",
        "",
        "The public deployments and their GitHub repositories are listed separately so that visitors can use an application immediately or inspect its accompanying materials and source code.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    owner = catalog["owner"]
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    known = catalog["apps"]
    repositories = [
        repo for repo in list_repositories(owner, token)
        if not repo.get("fork") and not repo.get("archived") and repo.get("default_branch")
    ]
    active: list[dict[str, str]] = []
    discovered = 0

    for repository in repositories:
        name = repository["name"]
        existing = known.get(name, {})
        try:
            paths = repository_tree(owner, repository, token)
        except RuntimeError as exc:
            print(f"Skipping {name}: {exc}", file=sys.stderr)
            continue
        sources = [(path, raw_file(owner, repository, path, token)) for path in relevant_paths(paths)]
        combined = "\n".join(text for _, text in sources)
        evidence = has_streamlit_implementation(sources)
        if not evidence and not existing:
            continue

        candidates: list[str] = []
        for candidate in [existing.get("app_url", ""), *normalized_urls(combined), guessed_url(name)]:
            if candidate and candidate not in candidates:
                candidates.append(candidate)
        live_url = next((candidate for candidate in candidates if is_live(candidate, token)), None)
        if not live_url:
            continue

        if not existing:
            discovered += 1
            existing = {
                "title": readable_title(name),
                "section": "unclassified",
                "description": "A newly discovered Streamlit application linked to this public GitHub project.",
            }
            known[name] = existing
        existing["app_url"] = live_url
        if existing.get("section") not in {section["id"] for section in catalog["sections"]}:
            existing["section"] = "unclassified"
        active.append({
            "repository": name,
            "repository_url": repository["html_url"],
            "title": existing["title"],
            "section": existing["section"],
            "description": existing["description"],
            "app_url": existing["app_url"],
        })

    CATALOG_PATH.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    PAGE_PATH.write_text(render(catalog, active), encoding="utf-8")
    print(f"Scanned {len(repositories)} public non-fork repositories; indexed {len(active)} live Streamlit apps; added {discovered} unclassified app(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
