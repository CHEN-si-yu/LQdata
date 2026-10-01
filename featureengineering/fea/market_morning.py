"""上午截止 11:30 的独立聚合；绝不使用下午成交量推断上午单位。"""
import numpy as np
import pandas as pd
from .derived import DerivedCache

class MorningLayer(DerivedCache):
    name = "market_morning"
    version = 2

    # ★ 2026-09-25：删掉本层的 `self.root` 覆写。它原先把缓存重定向到
    #   `CodeX/featureengineering_cache`，与 `DerivedCache` 里的同名逻辑重复
    #   （09-24 的补丁只改了父类、漏了这里，两处各写各的）。现在统一走父类的
    #   `data/derived/<name>/`，不再有第二份路径逻辑。
    #   `FEA_CACHE_DIR` 由父类处理，行为不变。

    def source_fingerprint(self, year):
        p = self.cfg.upstream / "stock_history_5min" / f"year={year}" / "data.parquet"
        if not p.exists(): return {"exists": False}
        st = p.stat()
        return {"exists": True, "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}

    @staticmethod
    def aggregate(df):
        if df.empty: return pd.DataFrame()
        df = df.copy()
        tt = df.trade_time.astype("string")
        hm = tt.str.slice(11, 16)
        df = df.loc[(hm >= "09:35") & (hm <= "11:30")].copy()
        if df.empty: return pd.DataFrame()
        expected={f"{m//60:02d}:{m%60:02d}" for m in range(575,691,5)}
        numeric=df[["open","high","low","close","amount"]].to_numpy(dtype=float)
        valid=np.isfinite(numeric).all(axis=1) & (numeric[:,:4]>0).all(axis=1) & (numeric[:,4]>=0)
        valid &= df.trade_time.astype("string").str.slice(11,16).isin(expected).to_numpy(bool)
        df=df.loc[valid].copy()
        if df.empty:return pd.DataFrame()
        df["high"]=df[["open","high","close"]].max(axis=1)
        df["low"]=df[["open","low","close"]].min(axis=1)
        df["trade_date"] = df.trade_time.astype("string").str.slice(0, 10)
        if df.duplicated(["stock_code", "trade_time"]).any():
            raise ValueError("上午分钟源含重复 (stock_code, trade_time)")
        df = df.sort_values(["stock_code", "trade_time"])
        g = df.groupby(["trade_date", "stock_code"], sort=True, observed=True)
        out = g.agg(am_open=("open", "first"), am_close=("close", "last"),
                    am_high=("high", "max"), am_low=("low", "min"),
                    am_amount=("amount", "sum"), bars=("close", "count"),
                    last_time=("trade_time", "last")).reset_index()
        # 缺 11:30 或缺任一棒时不把较早收盘冒充上午收盘；停牌/缺测保留缺失。
        complete = (out.bars == 24) & out.last_time.astype("string").str.slice(11,16).eq("11:30")
        columns = ["am_open", "am_close", "am_high", "am_low", "am_amount"]
        out.loc[~complete, columns] = np.nan
        return out[["trade_date", "stock_code", *columns, "bars"]]

    def build_year(self, year, up):
        src = self.cfg.upstream / "stock_history_5min" / f"year={year}" / "data.parquet"
        if not src.exists(): return pd.DataFrame()
        frames = []
        for month in range(1, 13):
            lo = f"{year}-{month:02d}-01"
            hi = f"{year+1}-01-01" if month == 12 else f"{year}-{month+1:02d}-01"
            df = pd.read_parquet(src, columns=["stock_code","trade_time","open","high","low","close","amount"],
                filters=[("trade_time",">=",lo),("trade_time","<",hi),
                         ("stock_code","in",self.codes.tolist())])
            part = self.aggregate(df)
            if not part.empty: frames.append(part)
        return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

    def panel(self, panel, field):
        y0,y1 = int(panel.dates[0])//10000, int(panel.dates[-1])//10000
        self.ensure(y0,y1)
        df = self._read_all(y0,y1)
        if df.empty: return panel.empty()
        from .dates import series_to_int
        pos = pd.Index(panel.codes).get_indexer(df.stock_code)
        good = pos >= 0
        return panel.place(pos[good], series_to_int(df.trade_date)[good],
                           df[field].to_numpy(dtype=float)[good])

def morning(ctx, field):
    if hasattr(ctx, "accessed"):ctx.accessed.add("stock_history_5min")
    layer = getattr(ctx.up, "_market_morning", None)
    if layer is None:
        layer = ctx.up._market_morning = MorningLayer(ctx.up,ctx.cfg,ctx.panel.codes)
    return layer.panel(ctx.panel, field)
