# AngelCV Scripts

This folder contains utility scripts to help with project maintenance and development workflows.

## 📋 Available Scripts

### `bump_version.py` - Version Management

A comprehensive script for managing project versions with semantic versioning.

**Features:**
- 🏷️ Reads latest Git tag automatically
- 📈 Supports semantic versioning (major/minor/patch)
- 📝 Updates `pyproject.toml` using `uv`
- 🎯 Interactive prompts for safety
- ✨ Beautiful CLI output with Rich
- 🔄 Handles Git commits and tags

**Usage:**

```bash
# Show current version status
python scripts/bump_version.py --current

# Bump patch version (1.2.3 -> 1.2.4)
python scripts/bump_version.py patch

# Bump minor version (1.2.3 -> 1.3.0)
python scripts/bump_version.py minor

# Bump major version (1.2.3 -> 2.0.0)
python scripts/bump_version.py major

# Or run directly (if executable)
./scripts/bump_version.py patch
```

**What it does:**
1. Gets the latest Git tag
2. Calculates the new version based on bump type
3. Shows you a preview of changes
4. Updates `pyproject.toml` with `uv version`
5. Optionally commits changes
6. Optionally creates a new Git tag
7. Reminds you to push changes to trigger CI/CD

## 🛠️ Best Practices

### Running Scripts
- Always run from project root: `python scripts/script_name.py`
- Scripts automatically detect project root location
- All scripts include help: `python scripts/script_name.py --help`

### Adding New Scripts
1. Create executable Python files in this folder
2. Use `#!/usr/bin/env python3` shebang
3. Include comprehensive docstrings
4. Add error handling and validation
5. Use Rich for beautiful output
6. Update this README

### Dependencies
Scripts should:
- Use standard library when possible
- Leverage existing project dependencies
- Gracefully handle missing dependencies
- Provide clear installation instructions

## 🚀 Integration with Workflow

The `bump_version.py` script integrates perfectly with the GitHub Actions release workflow:

1. **Bump version locally**: `python scripts/bump_version.py patch`
2. **Push changes**: `git push && git push --tags`
3. **Automatic release**: GitHub Actions triggers on tag push
4. **Result**: Package published to PyPI + GitHub Release created

## 💡 Tips

- Use `--current` flag to check version status before bumping
- The script prevents common mistakes with interactive prompts
- Git tags trigger the automated release pipeline
- Always review changes before confirming operations 