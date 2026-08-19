# SPDX-License-Identifier: Apache-2.0

import io
import platform

from vllm.distributed import nixl_utils as nu


def test_cpu_no_avx_skips_nixl(monkeypatch):
    # Simulate an x86 CPU without the 'avx' flag in /proc/cpuinfo.
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr("builtins.open", lambda path, *args, **kwargs: io.StringIO("processor\t: 0\nflags\t: fpu vme de\n"))

    # Ensure any cached value is removed
    nu_globals = nu.__dict__
    if "NixlWrapper" in nu_globals:
        del nu_globals["NixlWrapper"]

    val = nu._load_nixl_attr("NixlWrapper")
    assert val is None
    assert nu_globals.get("NixlWrapper") is None
