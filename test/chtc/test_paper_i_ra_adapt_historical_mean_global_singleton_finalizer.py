from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = (
    REPO_ROOT
    / "chtc"
    / "paper_i_ra_adapt_repair_20260727"
    / "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_r50_20260802_v2_chtc"
)


def test_worker_publishes_results_without_cross_device_directory_rename(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "temporary-filesystem" / "artifacts"
    staging.mkdir(parents=True)
    (staging / "result.json").write_text('{"status":"passed"}\n', encoding="utf-8")
    output = tmp_path / "condor-scratch" / "worker_outputs" / "artifacts"

    probe = r'''
import errno
import importlib.util
from pathlib import Path
import sys

package = Path(sys.argv[1])
staging = Path(sys.argv[2])
output = Path(sys.argv[3])
sys.path.insert(0, str(package))
spec = importlib.util.spec_from_file_location("finalizer_probe", package / "run_cell.py")
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
real_rename = module.os.rename

def reject_cross_device_source_rename(source, destination):
    if Path(source) == staging and Path(destination) == output:
        raise OSError(errno.EXDEV, "simulated cross-device directory rename")
    return real_rename(source, destination)

module.os.rename = reject_cross_device_source_rename
module._publish_staging_directory(staging, output)
assert (output / "result.json").read_text(encoding="utf-8") == '{"status":"passed"}\n'
'''
    completed = subprocess.run(
        [sys.executable, "-B", "-c", probe, str(PACKAGE_DIR), str(staging), str(output)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    worker_source = (PACKAGE_DIR / "run_cell.py").read_text(encoding="utf-8")
    assert "_publish_staging_directory(staging, output_dir)" in worker_source
    assert "os.rename(staging, output_dir)" not in worker_source
