# Robot Assets

## Bundled Go2 USD

`go2/go2.usd` is the Go2 USD asset used by the reported blind rough-terrain
AsymPPO candidate.

Expected naming contract:

```text
base body: base
foot/contact bodies: .*_foot
height scanner prim: {ENV_REGEX_NS}/Robot/base
```

The training configs use this bundled asset by default. Override it only if you
need to test a different Go2 description:

```bash
export GO2_USD_PATH=/path/to/custom/go2.usd
```

For custom USDs with `base_link` and no separate `*_foot` bodies:

```bash
export GO2_BASE_BODY_NAME=base_link
export GO2_FOOT_BODY_REGEX='.*_calf'
export GO2_HEIGHT_SCANNER_PRIM='{ENV_REGEX_NS}/Robot/base_link'
```
