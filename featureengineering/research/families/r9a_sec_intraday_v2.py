"""R9-A / unit A-2: stock exposure to its OWN sector's intraday realised volatility.

Frontier: `tdx_minute` (5-minute bars of TDX *board* indices, zero usage in the 177
factor catalog).  One deleted construct family is nearby and is named here explicitly
rather than hidden -- see `note` in the candidate spec for the differentiation argument.

Data facts measured 2026-10-05 (evidence: artifacts/research/round9_20261005/
a_chip_distribution/a_report.md):
  * tdx_minute: year partitions 2011..2026, 1121 distinct board_code, 48 bars/day
    (09:35..15:00), so a "day" is a complete 5-minute bar set.
  * `board_name` is used only as a join key, never as a numeric signal --
    conf/field_exclusions.json says exactly that for tdx_minute.board_name.
  * There is NO block->stock constituent table locally.  `tdx_blocks` (1121 rows:
    block_code/block_name/block_type) and `dc_blocks` (1031 rows) are classification
    dictionaries with no members, and BOTH fail `Upstream._audit_columns` inside a
    research run (no valid date column -> ValueError).  The live market factor
    `mkt_sector_dispersion` documents the same absence ("local has no historical
    constituent table").  Therefore the sector map is built from `stock_list.industry`,
    which IS audit-safe because `Upstream._audit_columns` whitelists `stock_list` as
    mode="contract" (a calendar/axis contract, not an information snapshot).
  * Measured map coverage: 2114/2114 frozen-pool stocks carry a non-empty industry and
    110/110 distinct industries resolve to a board_name present in tdx_minute.

Memory note: tdx_minute is read one/two year partitions at a time and rows are filtered
to the ~110 needed boards before any per-group work; the 60M-row table is never held
whole.  Arrow-backed string columns are kept as-is (no `.astype(str)` on large frames).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fea import mathx as mx
from fea.prices import day_ints

SIGNALS = ("sec_own_rv_beta", "sec_own_rv_beta_exrelabel")

# ---- constants frozen before the first run; never tuned on IC or returns ----
WINDOW = 60          # rolling beta window, trading days
MIN_COUNT = 54       # 90% of WINDOW must be present, else NaN
MIN_RET = 40         # a sector-day needs >= 40 valid 5-min returns out of 47
VAR_FLOOR = 1e-12    # guard against a degenerate regressor

# Frozen at registration time from stock_limit_list 2022+ labels: stocks carrying more than
# one distinct DC-style industry label inside the post-rename regime.  Used ONLY by the
# exrelabel sensitivity signal (Lead condition 1).  The list is part of the immutable
# implementation (source_hash), not a runtime parameter.
EXCLUDE_RELABELLED = frozenset(('000008.SZ', '000009.SZ', '000019.SZ', '000025.SZ', '000026.SZ', '000032.SZ', '000034.SZ', '000035.SZ', '000055.SZ', '000059.SZ', '000060.SZ', '000062.SZ', '000065.SZ', '000068.SZ', '000090.SZ', '000096.SZ', '000151.SZ', '000158.SZ', '000159.SZ', '000301.SZ', '000404.SZ', '000407.SZ', '000417.SZ', '000420.SZ', '000421.SZ', '000426.SZ', '000428.SZ', '000498.SZ', '000501.SZ', '000503.SZ', '000507.SZ', '000519.SZ', '000521.SZ', '000524.SZ', '000530.SZ', '000531.SZ', '000532.SZ', '000537.SZ', '000539.SZ', '000541.SZ', '000547.SZ', '000551.SZ', '000552.SZ', '000553.SZ', '000554.SZ', '000555.SZ', '000558.SZ', '000561.SZ', '000576.SZ', '000590.SZ', '000591.SZ', '000592.SZ', '000593.SZ', '000596.SZ', '000597.SZ', '000599.SZ', '000600.SZ', '000601.SZ', '000603.SZ', '000607.SZ', '000612.SZ', '000626.SZ', '000628.SZ', '000630.SZ', '000631.SZ', '000636.SZ', '000652.SZ', '000665.SZ', '000672.SZ', '000676.SZ', '000681.SZ', '000685.SZ', '000688.SZ', '000690.SZ', '000695.SZ', '000701.SZ', '000702.SZ', '000703.SZ', '000709.SZ', '000712.SZ', '000713.SZ', '000715.SZ', '000716.SZ', '000721.SZ', '000722.SZ', '000723.SZ', '000729.SZ', '000731.SZ', '000733.SZ', '000735.SZ', '000738.SZ', '000751.SZ', '000753.SZ', '000758.SZ', '000759.SZ', '000762.SZ', '000767.SZ', '000768.SZ', '000776.SZ', '000777.SZ', '000778.SZ', '000782.SZ', '000783.SZ', '000785.SZ', '000789.SZ', '000790.SZ', '000791.SZ', '000795.SZ', '000798.SZ', '000799.SZ', '000801.SZ', '000810.SZ', '000812.SZ', '000815.SZ', '000819.SZ', '000823.SZ', '000829.SZ', '000833.SZ', '000850.SZ', '000859.SZ', '000862.SZ', '000863.SZ', '000875.SZ', '000876.SZ', '000877.SZ', '000878.SZ', '000880.SZ', '000882.SZ', '000885.SZ', '000886.SZ', '000887.SZ', '000892.SZ', '000899.SZ', '000906.SZ', '000910.SZ', '000917.SZ', '000919.SZ', '000920.SZ', '000921.SZ', '000923.SZ', '000925.SZ', '000928.SZ', '000930.SZ', '000935.SZ', '000936.SZ', '000938.SZ', '000949.SZ', '000951.SZ', '000955.SZ', '000958.SZ', '000959.SZ', '000966.SZ', '000967.SZ', '000969.SZ', '000970.SZ', '000973.SZ', '000978.SZ', '000983.SZ', '000985.SZ', '000988.SZ', '000993.SZ', '000997.SZ', '000998.SZ', '001696.SZ', '001896.SZ', '002004.SZ', '002008.SZ', '002011.SZ', '002012.SZ', '002014.SZ', '002015.SZ', '002017.SZ', '002023.SZ', '002025.SZ', '002029.SZ', '002033.SZ', '002034.SZ', '002038.SZ', '002039.SZ', '002041.SZ', '002042.SZ', '002043.SZ', '002050.SZ', '002051.SZ', '002053.SZ', '002056.SZ', '002057.SZ', '002059.SZ', '002060.SZ', '002062.SZ', '002067.SZ', '002068.SZ', '002075.SZ', '002078.SZ', '002079.SZ', '002083.SZ', '002084.SZ', '002091.SZ', '002093.SZ', '002094.SZ', '002095.SZ', '002098.SZ', '002101.SZ', '002103.SZ', '002107.SZ', '002108.SZ', '002111.SZ', '002114.SZ', '002115.SZ', '002120.SZ', '002125.SZ', '002127.SZ', '002128.SZ', '002130.SZ', '002131.SZ', '002135.SZ', '002137.SZ', '002138.SZ', '002144.SZ', '002146.SZ', '002150.SZ', '002151.SZ', '002154.SZ', '002159.SZ', '002160.SZ', '002161.SZ', '002162.SZ', '002163.SZ', '002166.SZ', '002171.SZ', '002172.SZ', '002174.SZ', '002179.SZ', '002181.SZ', '002182.SZ', '002183.SZ', '002184.SZ', '002189.SZ', '002191.SZ', '002195.SZ', '002196.SZ', '002203.SZ', '002204.SZ', '002205.SZ', '002213.SZ', '002215.SZ', '002216.SZ', '002218.SZ', '002221.SZ', '002224.SZ', '002229.SZ', '002232.SZ', '002235.SZ', '002238.SZ', '002242.SZ', '002243.SZ', '002249.SZ', '002250.SZ', '002254.SZ', '002258.SZ', '002264.SZ', '002266.SZ', '002267.SZ', '002268.SZ', '002269.SZ', '002272.SZ', '002274.SZ', '002277.SZ', '002286.SZ', '002291.SZ', '002292.SZ', '002295.SZ', '002297.SZ', '002298.SZ', '002301.SZ', '002302.SZ', '002303.SZ', '002307.SZ', '002317.SZ', '002324.SZ', '002329.SZ', '002330.SZ', '002331.SZ', '002334.SZ', '002335.SZ', '002338.SZ', '002342.SZ', '002343.SZ', '002345.SZ', '002347.SZ', '002348.SZ', '002349.SZ', '002355.SZ', '002361.SZ', '002364.SZ', '002368.SZ', '002370.SZ', '002373.SZ', '002374.SZ', '002377.SZ', '002380.SZ', '002383.SZ', '002384.SZ', '002385.SZ', '002389.SZ', '002394.SZ', '002395.SZ', '002397.SZ', '002400.SZ', '002402.SZ', '002403.SZ', '002404.SZ', '002408.SZ', '002409.SZ', '002412.SZ', '002413.SZ', '002414.SZ', '002419.SZ', '002421.SZ', '002429.SZ', '002430.SZ', '002436.SZ', '002442.SZ', '002443.SZ', '002446.SZ', '002453.SZ', '002458.SZ', '002463.SZ', '002465.SZ', '002468.SZ', '002469.SZ', '002474.SZ', '002476.SZ', '002478.SZ', '002479.SZ', '002480.SZ', '002484.SZ', '002486.SZ', '002489.SZ', '002492.SZ', '002494.SZ', '002511.SZ', '002515.SZ', '002517.SZ', '002518.SZ', '002519.SZ', '002521.SZ', '002522.SZ', '002532.SZ', '002534.SZ', '002539.SZ', '002541.SZ', '002543.SZ', '002545.SZ', '002548.SZ', '002554.SZ', '002558.SZ', '002561.SZ', '002565.SZ', '002566.SZ', '002567.SZ', '002568.SZ', '002573.SZ', '002574.SZ', '002576.SZ', '002577.SZ', '002578.SZ', '002579.SZ', '002582.SZ', '002585.SZ', '002588.SZ', '002591.SZ', '002593.SZ', '002596.SZ', '002597.SZ', '002605.SZ', '002612.SZ', '002614.SZ', '002615.SZ', '002617.SZ', '002622.SZ', '002624.SZ', '002625.SZ', '002626.SZ', '002627.SZ', '002628.SZ', '002631.SZ', '002632.SZ', '002636.SZ', '002639.SZ', '002644.SZ', '002645.SZ', '002646.SZ', '002649.SZ', '002651.SZ', '002654.SZ', '002657.SZ', '002658.SZ', '002661.SZ', '002663.SZ', '002670.SZ', '002671.SZ', '002672.SZ', '002674.SZ', '002676.SZ', '002677.SZ', '002678.SZ', '002679.SZ', '002682.SZ', '002685.SZ', '002687.SZ', '002688.SZ', '002695.SZ', '002696.SZ', '002698.SZ', '002701.SZ', '002702.SZ', '002705.SZ', '002709.SZ', '002712.SZ', '002723.SZ', '002724.SZ', '002728.SZ', '002729.SZ', '002734.SZ', '002735.SZ', '002738.SZ', '002739.SZ', '002747.SZ', '002752.SZ', '002756.SZ', '002760.SZ', '002761.SZ', '002763.SZ', '002765.SZ', '002767.SZ', '002768.SZ', '002769.SZ', '002771.SZ', '002772.SZ', '002773.SZ', '002775.SZ', '002777.SZ', '002778.SZ', '002780.SZ', '002782.SZ', '002787.SZ', '002790.SZ', '002796.SZ', '002797.SZ', '002798.SZ', '002799.SZ', '002803.SZ', '002805.SZ', '002806.SZ', '002809.SZ', '002811.SZ', '002815.SZ', '002818.SZ', '002819.SZ', '002820.SZ', '002821.SZ', '002823.SZ', '002824.SZ', '002825.SZ', '002826.SZ', '002828.SZ', '002829.SZ', '002831.SZ', '002832.SZ', '002836.SZ', '002838.SZ', '002840.SZ', '002843.SZ', '002846.SZ', '002847.SZ', '002849.SZ', '002851.SZ', '002852.SZ', '002853.SZ', '002858.SZ', '002859.SZ', '002860.SZ', '002862.SZ', '002865.SZ', '002867.SZ', '002869.SZ', '002873.SZ', '002875.SZ', '002877.SZ', '002878.SZ', '002881.SZ', '002885.SZ', '002886.SZ', '002887.SZ', '002889.SZ', '002891.SZ', '002892.SZ', '002893.SZ', '002895.SZ', '002897.SZ', '002899.SZ', '002905.SZ', '002907.SZ', '002908.SZ', '002910.SZ', '002911.SZ', '002912.SZ', '002913.SZ', '002916.SZ', '002918.SZ', '002919.SZ', '002921.SZ', '002922.SZ', '600010.SH', '600011.SH', '600018.SH', '600021.SH', '600023.SH', '600052.SH', '600058.SH', '600059.SH', '600063.SH', '600066.SH', '600067.SH', '600072.SH', '600076.SH', '600088.SH', '600095.SH', '600096.SH', '600101.SH', '600103.SH', '600108.SH', '600113.SH', '600118.SH', '600123.SH', '600126.SH', '600127.SH', '600128.SH', '600129.SH', '600131.SH', '600133.SH', '600135.SH', '600137.SH', '600141.SH', '600143.SH', '600151.SH', '600152.SH', '600156.SH', '600158.SH', '600163.SH', '600166.SH', '600167.SH', '600172.SH', '600183.SH', '600184.SH', '600185.SH', '600188.SH', '600189.SH', '600192.SH', '600195.SH', '600199.SH', '600201.SH', '600206.SH', '600211.SH', '600217.SH', '600219.SH', '600222.SH', '600223.SH', '600227.SH', '600229.SH', '600231.SH', '600232.SH', '600233.SH', '600236.SH', '600237.SH', '600248.SH', '600249.SH', '600250.SH', '600251.SH', '600252.SH', '600256.SH', '600257.SH', '600261.SH', '600284.SH', '600288.SH', '600292.SH', '600293.SH', '600307.SH', '600308.SH', '600310.SH', '600313.SH', '600316.SH', '600326.SH', '600327.SH', '600330.SH', '600331.SH', '600333.SH', '600336.SH', '600338.SH', '600343.SH', '600353.SH', '600356.SH', '600359.SH', '600361.SH', '600362.SH', '600363.SH', '600366.SH', '600371.SH', '600389.SH', '600391.SH', '600395.SH', '600398.SH', '600400.SH', '600405.SH', '600410.SH', '600415.SH', '600418.SH', '600419.SH', '600422.SH', '600429.SH', '600433.SH', '600435.SH', '600436.SH', '600446.SH', '600448.SH', '600449.SH', '600452.SH', '600455.SH', '600458.SH', '600463.SH', '600469.SH', '600475.SH', '600478.SH', '600479.SH', '600480.SH', '600481.SH', '600482.SH', '600490.SH', '600495.SH', '600496.SH', '600497.SH', '600502.SH', '600505.SH', '600507.SH', '600508.SH', '600509.SH', '600512.SH', '600516.SH', '600517.SH', '600527.SH', '600528.SH', '600531.SH', '600536.SH', '600538.SH', '600551.SH', '600557.SH', '600562.SH', '600563.SH', '600567.SH', '600569.SH', '600572.SH', '600573.SH', '600575.SH', '600576.SH', '600578.SH', '600580.SH', '600583.SH', '600590.SH', '600594.SH', '600596.SH', '600598.SH', '600604.SH', '600605.SH', '600613.SH', '600616.SH', '600617.SH', '600619.SH', '600620.SH', '600621.SH', '600630.SH', '600633.SH', '600635.SH', '600637.SH', '600640.SH', '600641.SH', '600644.SH', '600655.SH', '600658.SH', '600662.SH', '600665.SH', '600667.SH', '600668.SH', '600673.SH', '600676.SH', '600679.SH', '600681.SH', '600682.SH', '600684.SH', '600685.SH', '600688.SH', '600689.SH', '600691.SH', '600693.SH', '600697.SH', '600706.SH', '600712.SH', '600718.SH', '600720.SH', '600724.SH', '600728.SH', '600737.SH', '600738.SH', '600740.SH', '600743.SH', '600744.SH', '600750.SH', '600751.SH', '600756.SH', '600758.SH', '600768.SH', '600769.SH', '600770.SH', '600773.SH', '600774.SH', '600779.SH', '600780.SH', '600783.SH', '600784.SH', '600785.SH', '600790.SH', '600792.SH', '600794.SH', '600796.SH', '600797.SH', '600800.SH', '600801.SH', '600802.SH', '600803.SH', '600810.SH', '600822.SH', '600824.SH', '600825.SH', '600826.SH', '600828.SH', '600838.SH', '600839.SH', '600841.SH', '600845.SH', '600846.SH', '600850.SH', '600854.SH', '600855.SH', '600857.SH', '600861.SH', '600862.SH', '600863.SH', '600864.SH', '600865.SH', '600868.SH', '600872.SH', '600874.SH', '600875.SH', '600876.SH', '600879.SH', '600880.SH', '600881.SH', '600882.SH', '600883.SH', '600884.SH', '600888.SH', '600893.SH', '600903.SH', '600909.SH', '600917.SH', '600936.SH', '600959.SH', '600961.SH', '600962.SH', '600963.SH', '600966.SH', '600967.SH', '600969.SH', '600975.SH', '600976.SH', '600977.SH', '600979.SH', '600980.SH', '600981.SH', '600982.SH', '600983.SH', '600985.SH', '600986.SH', '600990.SH', '600992.SH', '600995.SH', '600996.SH', '600999.SH', '601001.SH', '601003.SH', '601011.SH', '601015.SH', '601016.SH', '601019.SH', '601086.SH', '601101.SH', '601107.SH', '601108.SH', '601116.SH', '601117.SH', '601118.SH', '601127.SH', '601137.SH', '601139.SH', '601168.SH', '601199.SH', '601208.SH', '601212.SH', '601222.SH', '601226.SH', '601233.SH', '601238.SH', '601333.SH', '601336.SH', '601366.SH', '601368.SH', '601369.SH', '601388.SH', '601390.SH', '601515.SH', '601566.SH', '601579.SH', '601588.SH', '601595.SH', '601599.SH', '601600.SH', '601611.SH', '601619.SH', '601666.SH', '601669.SH', '601677.SH', '601699.SH', '601717.SH', '601727.SH', '601789.SH', '601808.SH', '601811.SH', '601857.SH', '601858.SH', '601888.SH', '601890.SH', '601898.SH', '601908.SH', '601929.SH', '601933.SH', '601949.SH', '601952.SH', '601969.SH', '601991.SH', '601992.SH', '601996.SH', '601999.SH', '603000.SH', '603002.SH', '603010.SH', '603015.SH', '603020.SH', '603025.SH', '603026.SH', '603031.SH', '603033.SH', '603038.SH', '603058.SH', '603066.SH', '603077.SH', '603079.SH', '603081.SH', '603086.SH', '603098.SH', '603101.SH', '603103.SH', '603116.SH', '603123.SH', '603126.SH', '603136.SH', '603138.SH', '603139.SH', '603165.SH', '603177.SH', '603180.SH', '603186.SH', '603196.SH', '603198.SH', '603200.SH', '603203.SH', '603208.SH', '603222.SH', '603223.SH', '603225.SH', '603226.SH', '603228.SH', '603258.SH', '603260.SH', '603266.SH', '603283.SH', '603303.SH', '603311.SH', '603313.SH', '603315.SH', '603316.SH', '603318.SH', '603319.SH', '603320.SH', '603326.SH', '603328.SH', '603329.SH', '603330.SH', '603336.SH', '603345.SH', '603355.SH', '603360.SH', '603365.SH', '603366.SH', '603369.SH', '603386.SH', '603393.SH', '603399.SH', '603416.SH', '603421.SH', '603444.SH', '603456.SH', '603466.SH', '603477.SH', '603496.SH', '603499.SH', '603500.SH', '603505.SH', '603508.SH', '603519.SH', '603527.SH', '603533.SH', '603535.SH', '603536.SH', '603558.SH', '603567.SH', '603578.SH', '603579.SH', '603585.SH', '603588.SH', '603598.SH', '603599.SH', '603601.SH', '603602.SH', '603607.SH', '603615.SH', '603616.SH', '603619.SH', '603630.SH', '603633.SH', '603636.SH', '603637.SH', '603660.SH', '603661.SH', '603665.SH', '603668.SH', '603677.SH', '603678.SH', '603686.SH', '603689.SH', '603690.SH', '603696.SH', '603698.SH', '603708.SH', '603711.SH', '603717.SH', '603722.SH', '603725.SH', '603726.SH', '603727.SH', '603728.SH', '603738.SH', '603776.SH', '603777.SH', '603778.SH', '603787.SH', '603797.SH', '603798.SH', '603801.SH', '603808.SH', '603817.SH', '603818.SH', '603819.SH', '603833.SH', '603839.SH', '603848.SH', '603859.SH', '603877.SH', '603878.SH', '603881.SH', '603886.SH', '603887.SH', '603888.SH', '603889.SH', '603898.SH', '603900.SH', '603903.SH', '603906.SH', '603912.SH', '603916.SH', '603917.SH', '603918.SH', '603919.SH', '603920.SH', '603928.SH', '603929.SH', '603933.SH', '603936.SH', '603937.SH', '603938.SH', '603955.SH', '603958.SH', '603960.SH', '603970.SH', '603978.SH', '603979.SH', '603985.SH', '603988.SH', '603989.SH', '603991.SH', '603993.SH', '603998.SH', '603999.SH'))

PARAMETERS = {"signal": "sec_own_rv_beta", "window": WINDOW, "min_count": MIN_COUNT}
PARAMETERS_EXREL = {"signal": "sec_own_rv_beta_exrelabel", "window": WINDOW, "min_count": MIN_COUNT}


def _sector_rv(tm: pd.DataFrame) -> pd.DataFrame:
    """(day x board_code) intraday realised vol = sqrt(sum of squared 5-min log returns).

    Bars are ordered by the ISO `trade_time` string, which sorts chronologically.
    The first bar of each day has no previous close and is dropped by `shift(1)`.
    """
    frame = tm.loc[:, ["board_code", "trade_time", "close"]].copy()
    frame["day"] = day_ints(frame["trade_time"].str.slice(0, 10))
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=np.float64)
    frame["close"] = close
    frame = frame.loc[np.isfinite(close) & (close > 0)].copy()
    frame = frame.sort_values(["board_code", "day", "trade_time"], kind="stable")
    grouped = frame.groupby(["board_code", "day"], sort=False)
    prev = grouped["close"].shift(1).to_numpy(dtype=np.float64)
    cur = frame["close"].to_numpy(dtype=np.float64)
    with np.errstate(all="ignore"):
        logret = np.log(cur / prev)
    frame["r2"] = np.where(np.isfinite(logret), logret * logret, np.nan)
    agg = frame.groupby(["board_code", "day"], sort=False)["r2"].agg(["sum", "count"])
    rv = np.sqrt(agg["sum"].to_numpy(dtype=np.float64))
    ok = agg["count"].to_numpy(dtype=np.float64) >= MIN_RET
    out = pd.DataFrame({"board_code": agg.index.get_level_values(0),
                        "day": agg.index.get_level_values(1).astype(np.int64),
                        "rv": np.where(ok, rv, np.nan)})
    return out.pivot(index="day", columns="board_code", values="rv")


def _board_lookup(tm: pd.DataFrame) -> dict:
    """board_name -> board_code, deterministic when a name matches several codes."""
    pairs = tm.loc[:, ["board_name", "board_code"]].dropna().drop_duplicates()
    pairs = pairs.assign(board_name=pairs["board_name"].astype(str).str.strip(),
                         board_code=pairs["board_code"].astype(str))
    weight = pairs.groupby(["board_name", "board_code"]).size().rename("n").reset_index()
    # most rows wins; ties broken by the lexicographically smallest board_code
    weight = weight.sort_values(["board_name", "n", "board_code"],
                                ascending=[True, False, True], kind="stable")
    best = weight.drop_duplicates("board_name", keep="first")
    return dict(zip(best["board_name"], best["board_code"]))


def _stock_board(ctx, board_of_name: dict, codes: np.ndarray) -> np.ndarray:
    """stock column -> board_code, via stock_list.industry == tdx_minute.board_name."""
    listing = ctx.dataset("stock_list", columns=["stock_code", "industry"])
    listing = listing.dropna(subset=["industry"])
    listing = listing.assign(stock_code=listing["stock_code"].astype(str),
                             industry=listing["industry"].astype(str).str.strip())
    listing = listing.loc[listing["industry"] != ""].drop_duplicates("stock_code", keep="last")
    industry = dict(zip(listing["stock_code"], listing["industry"]))
    return np.array([board_of_name.get(industry.get(str(code), ""), "") for code in codes],
                    dtype=object)


def build(parameters):
    if parameters not in (PARAMETERS, PARAMETERS_EXREL):
        raise ValueError("Only the two frozen R9-A sector-intraday definitions are permitted")
    drop_relabelled = parameters == PARAMETERS_EXREL

    def compute(ctx):
        days = np.asarray(ctx.panel.dates, dtype=np.int64)
        codes = np.asarray(ctx.panel.codes, dtype=object)
        shape = (days.size, codes.size)
        y0, y1 = int(days[0]) // 10000, int(days[-1]) // 10000
        tm = ctx.dataset("tdx_minute",
                         columns=["board_code", "board_name", "trade_time", "close"],
                         years=(y0, y1))
        if tm is None or tm.empty:
            return np.full(shape, np.nan, dtype=np.float64)

        stock_board = _stock_board(ctx, _board_lookup(tm), codes)
        needed = {b for b in stock_board if b}
        if not needed:
            return np.full(shape, np.nan, dtype=np.float64)
        tm = tm.loc[tm["board_code"].isin(needed)]
        table = _sector_rv(tm)
        del tm
        if table.empty:
            return np.full(shape, np.nan, dtype=np.float64)

        table = table.reindex(index=days)
        pos = {c: i for i, c in enumerate(table.columns)}
        idx = np.array([pos.get(b, -1) for b in stock_board], dtype=np.int64)
        values = table.to_numpy(dtype=np.float64)
        # -1 must stay NaN: indexing with -1 would silently take the last column.
        regressor = np.full(shape, np.nan, dtype=np.float64)
        ok = idx >= 0
        if ok.any():
            regressor[:, ok] = values[:, idx[ok]]

        if drop_relabelled:
            keep = np.array([str(c) not in EXCLUDE_RELABELLED for c in codes], dtype=bool)
            regressor = np.where(keep[None, :], regressor, np.nan)

        returns = np.asarray(ctx.ret(1), dtype=np.float64)
        with np.errstate(all="ignore"):
            covariance = mx.roll_cov(returns, regressor, WINDOW, min_count=MIN_COUNT)
            variance = mx.roll_var(regressor, WINDOW, min_count=MIN_COUNT)
            beta = np.where(variance > VAR_FLOOR, covariance / variance, np.nan)
        return np.where(np.isfinite(beta), beta, np.nan)

    return compute
