#!/usr/bin/env python3
"""
Version Bumping Script for AngelCV

This script helps you bump the version of the project by:
1. Getting the latest Git tag
2. Incrementing it (major/minor/patch)
3. Updating pyproject.toml with uv
4. Creating a new Git tag
5. Optionally committing changes

Usage:
    python scripts/bump_version.py patch
    python scripts/bump_version.py minor
    python scripts/bump_version.py major
    python scripts/bump_version.py --current  # Show current version
"""

import argparse
from pathlib import Path
import subprocess
import sys

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt
    from rich.text import Text
except ImportError:
    print("❌ Rich not found. Install with: uv add rich")
    sys.exit(1)

console = Console()


class VersionBumper:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.pyproject_path = project_root / "pyproject.toml"

    def run_command(self, cmd: list[str], capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run a command and return the result."""
        try:
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=capture_output, text=True, check=True)
            return result
        except subprocess.CalledProcessError as e:
            console.print(f"❌ Command failed: {' '.join(cmd)}", style="red")
            console.print(f"Error: {e.stderr}", style="red")
            sys.exit(1)
        except FileNotFoundError:
            console.print(f"❌ Command not found: {cmd[0]}", style="red")
            console.print("Make sure all required tools are installed.", style="yellow")
            sys.exit(1)

    def get_latest_tag(self) -> str | None:
        """Get the latest Git tag."""
        try:
            result = self.run_command(["git", "describe", "--tags", "--abbrev=0"])
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None

    def parse_version(self, version_str: str) -> tuple[int, int, int]:
        """Parse a version string into major, minor, patch components."""
        # Remove 'v' prefix if present
        clean_version = version_str.lstrip("v")

        try:
            parts = clean_version.split(".")
            if len(parts) != 3:
                raise ValueError("Version must have exactly 3 parts")

            major, minor, patch = map(int, parts)
            return major, minor, patch
        except (ValueError, TypeError) as e:
            console.print(f"❌ Invalid version format: {version_str}", style="red")
            console.print("Expected format: v1.2.3 or 1.2.3", style="yellow")
            sys.exit(1)

    def bump_version(self, current: str, bump_type: str) -> str:
        """Bump version based on type (major/minor/patch)."""
        major, minor, patch = self.parse_version(current)

        if bump_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif bump_type == "minor":
            minor += 1
            patch = 0
        elif bump_type == "patch":
            patch += 1
        else:
            console.print(f"❌ Invalid bump type: {bump_type}", style="red")
            sys.exit(1)

        return f"{major}.{minor}.{patch}"

    def get_current_pyproject_version(self) -> str | None:
        """Get current version from pyproject.toml."""
        try:
            result = self.run_command(["uv", "version"])
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None

    def update_pyproject_version(self, new_version: str) -> None:
        """Update version in pyproject.toml using uv."""
        console.print(f"📝 Updating pyproject.toml to version {new_version}...")
        self.run_command(["uv", "version", new_version])
        console.print("✅ pyproject.toml updated", style="green")

    def create_git_tag(self, version: str) -> None:
        """Create a new Git tag."""
        tag_name = f"v{version}"
        console.print(f"🏷️  Creating Git tag: {tag_name}...")
        self.run_command(["git", "tag", tag_name])
        console.print(f"✅ Git tag {tag_name} created", style="green")

    def commit_changes(self, version: str) -> None:
        """Commit the version changes."""
        console.print("📝 Committing changes...")
        self.run_command(["git", "add", "pyproject.toml"])
        self.run_command(["git", "commit", "-m", f"Bump version to {version}"])
        console.print("✅ Changes committed", style="green")

    def show_git_diff(self) -> None:
        """Show git diff for pyproject.toml."""
        try:
            result = self.run_command(["git", "diff", "pyproject.toml"])
            if result.stdout.strip():
                console.print("\n📋 Changes to be committed:", style="bold blue")
                console.print(Panel(result.stdout, title="git diff pyproject.toml", border_style="yellow"))
            else:
                console.print("📋 No changes detected in pyproject.toml", style="yellow")
        except subprocess.CalledProcessError:
            console.print("⚠️  Could not show git diff", style="yellow")

    def show_current_status(self) -> None:
        """Show current version status."""
        latest_tag = self.get_latest_tag()
        pyproject_version = self.get_current_pyproject_version()

        panel_content = []

        if latest_tag:
            panel_content.append(f"🏷️  Latest Git tag: {latest_tag}")
        else:
            panel_content.append("🏷️  No Git tags found")

        if pyproject_version:
            panel_content.append(f"📄 pyproject.toml version: {pyproject_version}")
        else:
            panel_content.append("📄 No version in pyproject.toml")

        console.print(Panel("\n".join(panel_content), title="📊 Current Version Status", border_style="blue"))

    def run_bump(self, bump_type: str) -> None:
        """Run the complete version bump process."""
        # Get current version
        latest_tag = self.get_latest_tag()
        if not latest_tag:
            console.print("❌ No Git tags found. Creating initial version...", style="yellow")
            current_version = "0.0.0"
        else:
            current_version = latest_tag

        # Calculate new version
        new_version = self.bump_version(current_version, bump_type)

        # Show what will happen
        console.print(
            Panel(
                f"Current version: {current_version}\nNew version: {new_version}\nBump type: {bump_type}",
                title="🚀 Version Bump Plan",
                border_style="green",
            )
        )

        # Confirm action
        if not Confirm.ask("Do you want to proceed?"):
            console.print("❌ Operation cancelled", style="yellow")
            return

        # Update pyproject.toml
        self.update_pyproject_version(new_version)

        # Show git diff
        self.show_git_diff()

        # Ask about committing
        if Confirm.ask("Do you want to commit the changes?"):
            self.commit_changes(new_version)

        # Ask about creating tag
        if Confirm.ask("Do you want to create a Git tag?"):
            self.create_git_tag(new_version)

        # Final status
        console.print(Panel(f"✅ Version bumped to {new_version}", title="🎉 Success!", border_style="green"))

        console.print("\n💡 Next steps:", style="bold blue")
        console.print("• Push changes: git push")
        console.print("• Push tags: git push --tags")
        console.print("• This will trigger the release workflow! 🚀")


def main():
    parser = argparse.ArgumentParser(
        description="Bump version for AngelCV project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/bump_version.py patch     # 1.2.3 -> 1.2.4
  python scripts/bump_version.py minor     # 1.2.3 -> 1.3.0
  python scripts/bump_version.py major     # 1.2.3 -> 2.0.0
  python scripts/bump_version.py --current # Show current version
        """,
    )

    parser.add_argument("bump_type", nargs="?", choices=["major", "minor", "patch"], help="Type of version bump")

    parser.add_argument("--current", action="store_true", help="Show current version status")

    args = parser.parse_args()

    # Find project root
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    # Validate we're in the right place
    if not (project_root / "pyproject.toml").exists():
        console.print("❌ pyproject.toml not found. Are you in the right directory?", style="red")
        sys.exit(1)

    bumper = VersionBumper(project_root)

    if args.current:
        bumper.show_current_status()
    elif args.bump_type:
        bumper.run_bump(args.bump_type)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
