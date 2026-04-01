# GitHub Action Setup

## Quick Start

1. Go to your repo's **Settings → Secrets and variables → Actions**
2. Add a secret called `OPENAI_API_KEY` with your OpenAI API key
3. Create `.github/workflows/pr-review.yml` in your repo:

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

4. Push and open a PR — the review will run automatically.

## Using a different LLM provider

Any OpenAI-compatible API works. Set `base_url` to your provider's endpoint:

```yaml
- uses: NoahLundSyrdal/prReviewer@v1
  with:
    api_key: ${{ secrets.LLM_API_KEY }}
    base_url: 'https://api.anthropic.com/v1'
    model: 'claude-3-sonnet'
```

## Excluding files

Skip generated files, lock files, or specific directories:

```yaml
- uses: NoahLundSyrdal/prReviewer@v1
  with:
    api_key: ${{ secrets.OPENAI_API_KEY }}
    exclude: '*.lock,*.min.js,dist/**,docs/**,*.generated.*'
```

## Single vs multi mode

- `single`: One balanced review pass. Faster, cheaper.
- `multi` (default): Three focused passes (correctness, security, performance), then deduplicates and merges. More thorough.

```yaml
- uses: NoahLundSyrdal/prReviewer@v1
  with:
    api_key: ${{ secrets.OPENAI_API_KEY }}
    mode: 'single'  # or 'multi'
```

## Review without posting comments

Set `post_comments: 'false'` to print the review in the action log without posting inline comments:

```yaml
- uses: NoahLundSyrdal/prReviewer@v1
  with:
    api_key: ${{ secrets.OPENAI_API_KEY }}
    post_comments: 'false'
```

## Secrets reference

| Secret | Required | Description |
|--------|----------|-------------|
| `OPENAI_API_KEY` (or any name) | Yes | Passed as `api_key` input |
| `GITHUB_TOKEN` | Auto | Automatically provided by GitHub Actions |

## Troubleshooting

- **No comments posted**: Check that `api_key` is set and the LLM provider is reachable. Look at the action logs.
- **Permission errors**: The workflow needs `pull-requests: write` in the `permissions` block.
- **Rate limited**: The action retries with backoff, but very large PRs with `multi` mode make 3x the API calls. Try `mode: 'single'` or increase `max_lines`.
- **Wrong model**: Make sure `model` matches what your `base_url` provider supports.
