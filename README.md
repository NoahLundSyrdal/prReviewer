# pr-reviewer

`pr-reviewer` is a product-grade CLI prototype for LLM-powered PR review.

It ingests a unified diff, runs structured analysis, and returns developer-first feedback with severity, category, confidence, and inline code-frame context. It can also post findings directly as inline comments on GitHub PRs or GitLab MRs.

## Why this exists

Most AI code-review demos are either vague or overbuilt. `pr-reviewer` optimizes for high signal and real workflow integration:

- grounded findings tied to visible diff hunks
- strict output schema and validation
- clean terminal UX with fast local loop
- optional PR/MR comment publishing on changed lines

## Feature highlights

- Review from patch file, stdin, or staged diff (`--cached`)
- Single-pass mode (`--mode single`) and multi-pass mode (`--mode multi`)
  - `multi` runs correctness, security, and performance passes, then dedupes and merges
- Automatic chunked review for large diffs so broad PRs keep more context than a single truncated excerpt
- Cross-chunk synthesis pass for large reviews so the final summary/verdict reflects the whole PR, not just isolated chunk outputs
- Structured findings with:
  - severity: `low|medium|high`
  - category: `bug|security|performance|maintainability`
  - confidence: `0.0-1.0`
  - exact hunk code-frame annotation
- Output formats: `text`, `markdown`, `json`
- Inline comment publishing:
  - GitHub PR review comments
  - GitLab MR discussions
- Robust fallback when LLM returns malformed JSON

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

After installation, you can use either `pr-reviewer ...` or `python -m pr_reviewer ...`.

`pr-reviewer` also supports repo-local defaults via `.pr-reviewer.toml` or `[tool.pr-reviewer]` in `pyproject.toml`.

Required environment variable:

```bash
export PR_REVIEWER_API_KEY="your_api_key"
```

Optional model/provider settings:

```bash
export PR_REVIEWER_BASE_URL="https://api.openai.com/v1"
export PR_REVIEWER_MODEL="gpt-4.1-mini"
```

## Demo TODO checklist (end-to-end)

Use this as a copy-paste runbook for a live demo.

- [ ] 1. Open repo and activate environment
  ```bash
  cd /Users/noahsyrdal/prReviewer
  python -m venv .venv
  source .venv/bin/activate
  pip install -e '.[dev]'
  ```

- [ ] 2. Configure API access
  ```bash
  export PR_REVIEWER_API_KEY="your_api_key"
  export PR_REVIEWER_BASE_URL="https://api.openai.com/v1"
  ```

- [ ] 3. Sanity check CLI
  ```bash
  pr-reviewer --help
  pr-reviewer review --help
  ```

- [ ] 4. Run built-in real-project demo patch (`travelSync`)
  ```bash
  python -m pr_reviewer review examples/travelsync_demo.patch --mode multi --format text --color always
  ```

- [ ] 5. Save shareable markdown output
  ```bash
  python -m pr_reviewer review examples/travelsync_demo.patch --mode multi --format markdown --save /tmp/pr-review-demo.md
  ```

- [ ] 6. Run on your own current project changes (staged diff)
  ```bash
  cd /path/to/your/project
  git add -p
  git diff --cached > /tmp/my_project_demo.patch
  cd /Users/noahsyrdal/prReviewer
  python -m pr_reviewer review /tmp/my_project_demo.patch --mode multi --format text --color always
  ```

- [ ] 7. Optional: dry-run inline PR comments (safe integration demo)
  ```bash
  python -m pr_reviewer review /tmp/my_project_demo.patch \
    --mode multi \
    --post github \
    --repo owner/repo \
    --pr 123 \
    --dry-run-post
  ```

- [ ] 8. Optional: post real inline comments
  ```bash
  export GITHUB_TOKEN="ghp_xxx"
  python -m pr_reviewer review /tmp/my_project_demo.patch \
    --mode multi \
    --post github \
    --repo owner/repo \
    --pr 123
  ```

## Core usage

Review a patch file:

```bash
python -m pr_reviewer review examples/travelsync_demo.patch --mode multi
```

Review current working diff:

```bash
git diff | python -m pr_reviewer review --stdin
```

Review staged changes:

```bash
python -m pr_reviewer review --cached
```

Run multi-pass review:

```bash
python -m pr_reviewer review examples/travelsync_demo.patch --mode multi
```

Compact terminal output:

```bash
python -m pr_reviewer review examples/travelsync_demo.patch --mode multi --compact --color always
```

Save markdown output:

```bash
python -m pr_reviewer review examples/travelsync_demo.patch --mode multi --format markdown --save review.md
```

## Config files

Config is discovered from the current working directory upward in this order:

1. `.pr-reviewer.toml`
2. `pyproject.toml` with `[tool.pr-reviewer]`

Explicit CLI flags override config values. Config values override environment/default values.

Example `.pr-reviewer.toml`:

```toml
model = "gpt-4.1-mini"
mode = "multi"
max_lines = 900
format = "markdown"
color = "always"
compact = false
```

Example `pyproject.toml`:

```toml
[tool.pr-reviewer]
mode = "multi"
max_lines = 900
color = "always"
```

You can also point directly at a config file:

```bash
pr-reviewer --config /path/to/.pr-reviewer.toml review --cached
```

## Posting findings to PR/MR

Only findings mapped to changed lines are posted.

GitHub PR comments:

```bash
export GITHUB_TOKEN="ghp_xxx"
python -m pr_reviewer review --cached \
  --mode multi \
  --post github \
  --repo owner/repo \
  --pr 123
```

GitLab MR comments:

```bash
export GITLAB_TOKEN="glpat-xxx"
python -m pr_reviewer review --cached \
  --mode multi \
  --post gitlab \
  --repo group/project \
  --mr 42
```

Dry run posting:

```bash
python -m pr_reviewer review --cached --post github --repo owner/repo --pr 123 --dry-run-post
```

## Strong demo assets

- Real project demo diff (travelSync): [`examples/travelsync_demo.patch`](./examples/travelsync_demo.patch)
- Real project demo output (terminal): [`examples/travelsync_demo_output.txt`](./examples/travelsync_demo_output.txt)

## CLI synopsis

```text
pr-reviewer [--config FILE] review [patch] [--stdin] [--cached]
                                    [--mode single|multi]
                                    [--model MODEL]
                                    [--max-lines N]
                                    [--format text|json|markdown]
                                    [--save FILE]
                                    [--compact]
                                    [--base-url URL]
                                    [--color auto|always|never]
                                    [--post github|gitlab]
                                    [--repo REPO]
                                    [--pr N]
                                    [--mr N]
                                    [--integration-token TOKEN]
                                    [--integration-base-url URL]
                                    [--dry-run-post]
```

`--max-lines` is the approximate per-request chunk budget. Large diffs are automatically split across multiple review calls and merged back into one result.

## Project layout

```text
pr_reviewer/
  cli.py          # command UX and orchestration
  parsing.py      # diff stats, hunk parsing, code-frame extraction
  reviewer.py     # prompt strategy, single/multi pass pipeline, dedupe
  llm.py          # provider abstraction + OpenAI-compatible client
  integrations.py # GitHub/GitLab inline comment publishing
  formatters.py   # text / markdown / json rendering
  models.py       # typed schema
```

## Testing

```bash
pytest -q
```

## Known limitations

- Input is still a unified diff: chunking and cross-chunk synthesis give a coherent summary across a large patch, but the model is not given the wider codebase (imports, callers, types outside the hunk).
- Provider/model quality affects finding quality.
- Some platform APIs may reject comments if diff position changed server-side.

## Next iteration ideas

- Repo-local config exists for CLI defaults (`.pr-reviewer.toml`); extend with explicit policy/rule text or pack files the prompts must follow.
- Add GitHub Checks / GitLab pipeline summary mode.
- Add consensus mode (compare two models, merge intersection).
