# pr-reviewer

[![CI](https://github.com/NoahLundSyrdal/prReviewer/actions/workflows/ci.yml/badge.svg)](https://github.com/NoahLundSyrdal/prReviewer/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

LLM-powered code review that posts structured, actionable inline comments on your pull requests — automatically.

## Add to your repo in 30 seconds

1. Add your OpenAI API key as a repo secret: **Settings → Secrets → Actions → `OPENAI_API_KEY`**

2. Create `.github/workflows/pr-review.yml`:

```yaml
name: PR Review
on:
  pull_request:
    types: [opened, synchronize]
permissions:
  contents: read
  pull-requests: write
jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: NoahLundSyrdal/prReviewer@v1
        with:
          api_key: ${{ secrets.OPENAI_API_KEY }}
```

3. Open a pull request. That's it.

## What you get

Every PR gets reviewed with structured findings:

- **Severity**: `low` | `medium` | `high`
- **Category**: `bug` | `security` | `performance` | `maintainability`
- **Confidence**: `0.0 – 1.0`
- **Inline code context**: exact hunk annotation with suggested fixes

Findings are posted as inline review comments directly on the changed lines.

## Customize

```yaml
- uses: NoahLundSyrdal/prReviewer@v1
  with:
    api_key: ${{ secrets.OPENAI_API_KEY }}
    mode: 'multi'              # 'single' (faster) or 'multi' (deeper: correctness + security + performance)
    model: 'gpt-4.1-mini'     # any OpenAI-compatible model
    base_url: 'https://api.openai.com/v1'  # or any compatible provider
    max_lines: '1200'          # diff chunk budget per LLM call
    exclude: '*.lock,docs/**'  # glob patterns to skip (comma-separated)
    post_comments: 'true'      # set to 'false' to just print the review without posting
```

## Action inputs

| Input | Required | Default | Description |
|-------|----------|---------|-------------|
| `api_key` | Yes | — | API key for OpenAI (or compatible provider) |
| `github_token` | No | `${{ github.token }}` | Token for posting review comments |
| `model` | No | `gpt-4.1-mini` | LLM model identifier |
| `mode` | No | `multi` | `single` (fast) or `multi` (deep, multi-pass) |
| `base_url` | No | `https://api.openai.com/v1` | OpenAI-compatible API base URL |
| `max_lines` | No | `1200` | Max diff lines per review chunk |
| `exclude` | No | — | Comma-separated glob patterns to skip |
| `post_comments` | No | `true` | Whether to post inline PR comments |

## How it works

1. Fetches the PR diff via GitHub API
2. Filters out excluded files
3. Splits large diffs into reviewable chunks
4. Sends each chunk through the LLM with a structured review prompt
5. In `multi` mode, runs separate correctness, security, and performance passes, then dedupes
6. Synthesizes findings across chunks into a coherent summary
7. Posts findings as inline review comments on the changed lines

## CLI usage

You can also use `pr-reviewer` as a standalone CLI tool:

```bash
pip install git+https://github.com/NoahLundSyrdal/prReviewer.git
export PR_REVIEWER_API_KEY="your-api-key"

# Review a diff
git diff | pr-reviewer review --stdin --mode multi

# Review staged changes
pr-reviewer review --cached --mode multi

# Review a patch file and save markdown output
pr-reviewer review changes.patch --format markdown --save review.md

# Post comments to a GitHub PR
pr-reviewer review changes.patch --post github --repo owner/repo --pr 123
```

Run `pr-reviewer --help` for all options.

## Config files

`pr-reviewer` supports repo-local defaults via `.pr-reviewer.toml` or `[tool.pr-reviewer]` in `pyproject.toml`:

```toml
# .pr-reviewer.toml
model = "gpt-4.1-mini"
mode = "multi"
max_lines = 900
format = "markdown"
color = "always"
```

CLI flags override config values. Config values override environment defaults.

## Known limitations

- The model only sees the diff, not the full codebase (imports, callers, types outside the hunk)
- LLM quality varies by model and provider
- Some platform APIs may reject comments if the diff position changed since the review ran

## Roadmap

- Custom review rules via `.pr-reviewer.toml` policy/rule text
- GitHub Checks / GitLab pipeline summary mode
- Consensus mode (compare two models, merge intersection)
- GitLab CI integration

## Development

```bash
git clone https://github.com/NoahLundSyrdal/prReviewer.git
cd prReviewer
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest -v
ruff check .
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[MIT](LICENSE)
