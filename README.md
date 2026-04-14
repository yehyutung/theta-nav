# theta-nav

## Virtual environment

- Python virtual environment: `.venv`
- Create it with: `python3 -m venv .venv`
- Activate it on macOS or Linux with: `. .venv/bin/activate`
- Upgrade pip inside the environment with: `python -m pip install --upgrade pip`
- MiniWorld is installed in the environment with: `python -m pip install MiniWorld`

## Git setup

- This folder is initialized as a Git repository on the `main` branch.
- The local repository is ready for commits once files are added and staged.
- A GitHub remote has not been attached yet because GitHub authentication is not available in this session.
- To link the repository later, add the remote with `git remote add origin <github-repo-url>` or authenticate with `gh auth login` and create the repo with `gh repo create`.

## Commit and push flow

1. Activate the virtual environment: `. .venv/bin/activate`
2. Check changes: `git status`
3. Stage files: `git add .`
4. Commit: `git commit -m "Initial project setup"`
5. Push after a remote is configured: `git push -u origin main`
