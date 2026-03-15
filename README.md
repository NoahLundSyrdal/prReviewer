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
pip install -r requirements.txt
```

Required environment variable:

```bash
export PR_REVIEWER_API_KEY="your_api_key"
```

Optional model/provider settings:

```bash
export PR_REVIEWER_BASE_URL="https://api.openai.com/v1"
export PR_REVIEWER_MODEL="gpt-4.1-mini"
```

## Core usage

Review a patch file:

```bash
python -m pr_reviewer review examples/sample.diff
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
python -m pr_reviewer review examples/sample.diff --mode multi
```

Compact terminal output:

```bash
python -m pr_reviewer review examples/sample.diff --compact --color always
```

Save markdown output:

```bash
python -m pr_reviewer review examples/sample.diff --format markdown --save review.md
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

- Sample diff: [`examples/sample.diff`](./examples/sample.diff)
- Sample output: [`examples/sample_output.md`](./examples/sample_output.md)

## CLI synopsis

```text
python -m pr_reviewer review [patch] [--stdin] [--cached]
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

- Reviews only visible diff context, not full repository semantics
- Provider/model quality affects finding quality
- Some platform APIs may reject comments if diff position changed server-side

## Next iteration ideas

- Add local policy/rule packs per repo
- Add GitHub Checks / GitLab pipeline summary mode
- Add consensus mode (compare two models, merge intersection)
# prReviewer
