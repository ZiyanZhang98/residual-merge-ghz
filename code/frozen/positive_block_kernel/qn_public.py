"""Public-release switches (not part of the original Snellius code).

The research code ran only on Snellius compute nodes and accepted an earlier artifact only if the *entire*
source inventory (sha256 of every .py/.json, keyed by repository path) was unchanged since the artifact was
written. In this public tree the paths differ, a few unrelated research files are omitted, and large binaries
(labels.pt, data.pt) are not distributed, so by default:
  * the Slurm/host guard is off;
  * source-inventory equality is not required (settings and the content hashes of every distributed
    artifact file are still verified);
  * artifact files that are listed in a manifest but not distributed are skipped.
Set QN_STRICT_PROVENANCE=1 to restore the original behaviour (and QN_STRICT_SLURM=1 for the host guard).
"""
import os
from pathlib import Path

STRICT = os.environ.get('QN_STRICT_PROVENANCE') == '1'
STRICT_SLURM = os.environ.get('QN_STRICT_SLURM') == '1'


def same_sources(recorded, current):
    return recorded == current if STRICT else True


def present(path):
    return STRICT or Path(path).exists()


def job_id():
    return os.environ.get('SLURM_JOB_ID', 'local')
