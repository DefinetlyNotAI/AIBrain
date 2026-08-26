# Wiki publishing

The repository workflow at `.github/workflows/publish-wiki.yml` mirrors the `docs/` folder into the GitHub wiki for the repository that runs it. It is repository-relative: when used in `DefinetlyNotAI/AIBrain`, its target is `https://github.com/DefinetlyNotAI/AIBrain.wiki.git`; no repository name is hard-coded into the workflow.

## Enable it

1. Create or enable the repository wiki in GitHub's repository settings.
2. Push this workflow to the `main` branch.
3. Change a file in `docs/` or use **Actions → Publish documentation to Wiki → Run workflow**.

GitHub creates the wiki Git repository only after the wiki has been enabled. A publish run before then will fail at `git clone`; enable the wiki once and rerun the workflow.

## What it publishes

The workflow checks out the source repository, clones the wiki using the Actions token, and synchronizes `docs/` into the wiki root. It deletes wiki files that no longer exist under `docs/`, then commits only when content changed.

`Home.md` becomes the wiki home page and `_Sidebar.md` provides wiki navigation. Keep both files in `docs/` and use extensionless links in `_Sidebar.md`; GitHub Wiki resolves those links to wiki pages.

## Permissions and safety

The workflow requests `contents: write` because it must push to the wiki Git repository. It uses the built-in short-lived `github.token`, not a long-lived personal access token. If organization policy restricts the default Actions token, grant the workflow write permission for repository contents or use an approved repository automation token with equivalent wiki-write access.

The action only runs automatically for `docs/` or its workflow-file changes on `main`; it can also be dispatched manually. It does not publish from pull requests.
