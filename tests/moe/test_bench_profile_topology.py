from __future__ import annotations

import json

from freetoken.moe.bench_profile import (
    CostCurvePoint,
    TopologyIdentity,
    load_topology_profile,
    topology_profile_document,
)


def _identity(**overrides):
    values = dict(
        pci_bdf="0000:41:00.0",
        numa_node=0,
        pcie_generation=3,
        pcie_width=16,
        gpu_name="NVIDIA TITAN RTX",
        compute_capability="7.5",
        driver_version="590.1",
        runtime_version="13.0",
        tensor_layout="ds_fp4:i512",
        concurrency=4,
    )
    values.update(overrides)
    return TopologyIdentity(**values)


def test_profile_key_distinguishes_same_gpu_name_on_different_links():
    assert _identity().key() != _identity(pci_bdf="0000:82:00.0", pcie_width=4).key()


def test_exact_topology_profile_round_trip(tmp_path):
    identity = _identity()
    doc = topology_profile_document(
        identity,
        h2d=(CostCurvePoint(payload_bytes=1024, duration_us=3.5),),
    )
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(doc))
    assert load_topology_profile(identity, str(path)) == doc
    assert load_topology_profile(_identity(pcie_width=8), str(path)) is None
