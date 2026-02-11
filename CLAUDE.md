# CLAUDE.md

## Git

Do **not** perform any git operations (commit, push, pull, checkout, etc.) unless explicitly asked. This includes staging files. The developer manages version control manually.

## Command Execution

Do **not** run commands in the background. All commands should run in the foreground so the developer can observe output in real time.

## Why

This is a scientific software project under highly interactive development. The developer is actively experimenting with different generation options, research approaches, and configurations. Manual control over version control and command execution is essential to avoid disrupting ongoing experiments or losing track of intermediate states.
