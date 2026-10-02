"""Check the clone bootstrap without downloading a Python environment."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("explicit_paths", [False, True])
def test_bootstrap_handles_clone_and_artifact_paths_with_spaces(tmp_path, explicit_paths):
    clone = tmp_path / "research checkout"
    (clone / "scripts").mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "scripts" / "bootstrap.sh"
    shutil.copyfile(source, clone / "scripts" / "bootstrap.sh")
    fake_bin = tmp_path / "fake executables"
    fake_bin.mkdir()
    record = tmp_path / "bootstrap record.json"
    # uv and the generated entry point record what a real external installation
    # would receive. No Python package or dataset is downloaded by this check.
    doctor_script = (
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "p = os.environ['BOOTSTRAP_RECORD']\n"
        "d = json.load(open(p))\n"
        "d['doctor_args'] = sys.argv[1:]\n"
        "json.dump(d, open(p, 'w'))\n"
    )
    (fake_bin / "uv").write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "env = pathlib.Path(os.environ['UV_PROJECT_ENVIRONMENT'])\n"
        "(env / 'bin').mkdir(parents=True)\n"
        "entry = env / 'bin' / 'firerisk'\n"
        f"entry.write_text({doctor_script!r})\n"
        "entry.chmod(0o755)\n"
        "json.dump({'args': sys.argv[1:], 'cwd': os.getcwd(),"
        " 'artifacts': os.environ['FIRERISK_HOME'],"
        " 'cache': os.environ['UV_CACHE_DIR'], 'environment': str(env)},"
        " open(os.environ['BOOTSTRAP_RECORD'], 'w'))\n"
    )
    (fake_bin / "uv").chmod(0o755)
    environment = os.environ.copy()
    for key in ["FIRERISK_HOME", "UV_CACHE_DIR", "UV_PROJECT_ENVIRONMENT"]:
        environment.pop(key, None)
    environment.update(
        PATH=f"{fake_bin}{os.pathsep}{environment['PATH']}",
        BOOTSTRAP_RECORD=str(record),
    )
    artifact_root = clone / ".artifacts"
    virtual_environment = clone / ".venv"
    cache_root = artifact_root / "uv-cache"
    if explicit_paths:
        artifact_root = tmp_path / "chosen artifacts"
        virtual_environment = tmp_path / "chosen environment"
        cache_root = tmp_path / "chosen cache"
        environment.update(
            FIRERISK_HOME=str(artifact_root),
            UV_CACHE_DIR=str(cache_root),
            UV_PROJECT_ENVIRONMENT=str(virtual_environment),
        )
    completed = subprocess.run(
        ["bash", str(clone / "scripts" / "bootstrap.sh")],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(record.read_text())
    assert observed == {
        "args": ["sync", "--frozen", "--extra", "dev", "--python", "3.12"],
        "cwd": str(clone),
        "artifacts": str(artifact_root),
        "cache": str(cache_root),
        "environment": str(virtual_environment),
        "doctor_args": ["doctor"],
    }
    assert artifact_root.is_dir()
    assert "Activate with:" in completed.stdout
