# Grid-aligned market-share sanity check

设置：Adaptive target=0.5、gamma=0，对手为 PersistentMarketMaker(0.4, 0.4, 0)，1 seed、10,000 steps。Adaptive algorithm、0.2 grid、EMA 和 cold start 均未修改。

| Persistent epsilon | Adaptive market share | Mean base epsilon |
|---:|---:|---:|
| 0.4 (aligned) | 0.7516 | 0.3913 |
| 0.5 (Phase 3 baseline) | 0.6606 | 0.6611 |

Aligned run 中 base epsilon 等于 0.4 的频率为 0.9859，最终 quote 相对 base 的 mean one-side skew 为 0.2161。

## 判断

Step 1 sanity check 通过：base 有 98.59% 的时间选择 grid-aligned 0.4。同时复核 previous_market_volume、gross-volume response 和 update timing，没有发现实现错误。

Aligned case 的整体 realized share 为 0.7516，并未比 misaligned baseline 更接近 0.5；原因是该指标包含 Step 2。非零库存时一侧 quote 被降低，mean one-side skew 为 0.2161，因此最终成交份额高于 symmetric 50/50 tie。
最终 share 不能单独作为 Step 1 的 correctness test。
因此 grid alignment 修复了 Step 1 的 base selection，但不会消除 Step 2 对 aggregate market share 的影响；本 sanity check 到此停止，不修改 grid。
