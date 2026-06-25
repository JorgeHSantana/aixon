import os
import pytest
from click.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


def test_new_creates_project_directory(runner, tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        from aixon.cli import app
        result = runner.invoke(app, ["new", "my-project"])
        assert result.exit_code == 0
        assert os.path.isdir("my-project")


def test_new_creates_agents_package(runner, tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        from aixon.cli import app
        runner.invoke(app, ["new", "my-project"])
        assert os.path.isdir("my-project/agents")
        assert os.path.isfile("my-project/agents/__init__.py")


def test_new_creates_example_agent(runner, tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        from aixon.cli import app
        runner.invoke(app, ["new", "my-project"])
        assert os.path.isfile("my-project/agents/greeter.py")
        content = open("my-project/agents/greeter.py").read()
        assert "LLMAgent" in content
        assert "GreeterAgent" in content


def test_new_creates_main_py(runner, tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        from aixon.cli import app
        runner.invoke(app, ["new", "my-project"])
        assert os.path.isfile("my-project/main.py")
        content = open("my-project/main.py").read()
        assert "autodiscover" in content
        assert "Server" in content
        assert "uvicorn" in content


def test_new_creates_pyproject_toml(runner, tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        from aixon.cli import app
        runner.invoke(app, ["new", "my-project"])
        assert os.path.isfile("my-project/pyproject.toml")
        content = open("my-project/pyproject.toml").read()
        assert "my-project" in content
        # Depends on the server stack via the aixon[server] extra (which itself
        # provides uvicorn[standard]) — assert the real dependency, not a comment.
        assert "aixon[server,cli]" in content
        # tomllib must parse it (a real, valid pyproject, not just substrings).
        import tomllib
        assert tomllib.loads(content)["project"]["name"] == "my-project"


def test_new_agents_init_contains_no_import_loop(runner, tmp_path):
    """The agents/__init__.py must be a bare marker — no circular imports."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        from aixon.cli import app
        runner.invoke(app, ["new", "my-project"])
        content = open("my-project/agents/__init__.py").read()
        # Must not contain 'import' of the agents submodules themselves
        assert "from agents" not in content
        assert "import agents" not in content


def test_new_fails_if_directory_exists(runner, tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        os.makedirs("existing-project")
        from aixon.cli import app
        result = runner.invoke(app, ["new", "existing-project"])
        assert result.exit_code != 0
        assert "already exists" in result.output or "already exists" in (result.stderr or "")


def test_new_prints_quickstart_instructions(runner, tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        from aixon.cli import app
        result = runner.invoke(app, ["new", "my-project"])
        assert result.exit_code == 0
        # Should tell the user how to get started
        assert "cd" in result.output
        assert "pip install" in result.output
