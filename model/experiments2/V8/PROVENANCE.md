# V8 package provenance

- run.py: V8-specific fixed-rule backtest and reporting.
- account_engine.py: frozen local copy of the shared audited account engine packaged with experiments2/V4.
- panel_loader.py: byte-for-byte copy of experiments2/V1/model.py, used only to load the existing four-bank price/factor panel.
- V8 does not modify V1–V7, retrain models, or depend on temporary data/tmp/vlines code. The SMA report is a written prior-research reference.
- Panel-loader SHA-256: cb1e0d3d437773d7e3f7f83891001122d592fc20264f487276377a32a55f3f9b
