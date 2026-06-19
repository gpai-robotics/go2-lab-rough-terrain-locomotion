"""Robot USD naming contract helpers.

The default config targets IsaacLab's Go2 asset:

- base body: ``base``
- foot/contact bodies: ``.*_foot``
- height-scanner prim: ``{ENV_REGEX_NS}/Robot/base``

Some Go2 USDs exported from other projects use ``base_link`` and terminate the
leg at calf bodies without separate foot links. Those assets can be used by
setting:

.. code-block:: bash

   export GO2_BASE_BODY_NAME=base_link
   export GO2_FOOT_BODY_REGEX='.*_calf'
   export GO2_HEIGHT_SCANNER_PRIM='{ENV_REGEX_NS}/Robot/base_link'
"""

from __future__ import annotations

import os
from pathlib import Path


def bundled_go2_usd_path() -> Path:
    return Path(__file__).resolve().parents[2] / "assets" / "robots" / "go2" / "go2.usd"


def go2_usd_path() -> str:
    return os.environ.get("GO2_USD_PATH", str(bundled_go2_usd_path()))


def base_body_name() -> str:
    return os.environ.get("GO2_BASE_BODY_NAME", "base")


def foot_body_regex() -> str:
    return os.environ.get("GO2_FOOT_BODY_REGEX", ".*_foot")


def height_scanner_prim_path() -> str:
    return os.environ.get("GO2_HEIGHT_SCANNER_PRIM", f"{{ENV_REGEX_NS}}/Robot/{base_body_name()}")


def print_asset_contract() -> None:
    print(
        "[GO2_ASSET_CONTRACT] "
        f"base_body={base_body_name()} "
        f"foot_body_regex={foot_body_regex()} "
        f"height_scanner_prim={height_scanner_prim_path()}"
    )
