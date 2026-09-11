# Notes for agents working in this repository

## Git commit identity

All commits in this repository — including everything under `site/` — must be
authored and committed as **cristian.malaia@gmail.com** (the repo's configured
`user.email`). Do not override it with `hello@brainiacginger.com` or any other
address.

Why: Vercel deploys resonate.page from this repository and only builds pushes
whose commit author is a member of the Vercel team. Commits authored as
hello@brainiacginger.com are pushed to GitHub but Vercel skips them, so the
live site does not update.
