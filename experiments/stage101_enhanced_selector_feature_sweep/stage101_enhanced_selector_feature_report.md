# Stage101 Enhanced Selector Feature Sweep

## Configuration

- train tasks: `96`
- eval tasks: `60`
- train examples: `589824`
- full feature dim: `75`
- keep fraction: `0.1`
- feature modes: `stage100_base, gap_endpoint_norms, gap_endpoint_rank`
- objectives: `topk_bce, energy_regression`
- no rendering, checkpoint, or heavy tensor output

## Summary

| feature mode | objective | candidate | base | gap | tasks | precision@keep | energy recall | oracle recall | relative recall |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| endpoint_only | endpoint_diff_baseline | endpoint_diff_baseline | linear | 4 | 23 | 0.3100568529056466 | 0.2578457370400429 | 0.6033855663693469 | 0.4116023273571678 |
| endpoint_only | endpoint_diff_baseline | endpoint_diff_baseline | linear | 8 | 19 | 0.3119484817511157 | 0.27749040722846985 | 0.6419753777353387 | 0.4117828874211562 |
| endpoint_only | endpoint_diff_baseline | endpoint_diff_baseline | linear | 16 | 18 | 0.20097063968165052 | 0.2058291733264923 | 0.5684697247213788 | 0.3390035058061282 |
| endpoint_only | endpoint_diff_baseline | endpoint_diff_baseline | stage65_adapter | 4 | 23 | 0.29103069072184357 | 0.13412074485550757 | 0.23996665620285532 | 0.5757957541424296 |
| endpoint_only | endpoint_diff_baseline | endpoint_diff_baseline | stage65_adapter | 8 | 19 | 0.27700831074463694 | 0.13563174087750285 | 0.2529745478379099 | 0.5572832179696936 |
| endpoint_only | endpoint_diff_baseline | endpoint_diff_baseline | stage65_adapter | 16 | 18 | 0.1901941259081165 | 0.11971673224535254 | 0.2512796148657799 | 0.49458976917796665 |
| gap_endpoint_norms | energy_regression | mlp_selector | linear | 4 | 23 | 0.3441105024970096 | 0.2922107130289078 | 0.6033855663693469 | 0.46823527761127637 |
| gap_endpoint_norms | energy_regression | mlp_selector | linear | 8 | 19 | 0.36720735462088333 | 0.32395490298145696 | 0.6419753777353387 | 0.4845810360030124 |
| gap_endpoint_norms | energy_regression | mlp_selector | linear | 16 | 18 | 0.257852532590429 | 0.25045521681507427 | 0.5684697247213788 | 0.41153565877013737 |
| gap_endpoint_norms | energy_regression | mlp_selector | stage65_adapter | 4 | 23 | 0.4050579181183939 | 0.16150322491708008 | 0.23996665620285532 | 0.686532665853915 |
| gap_endpoint_norms | energy_regression | mlp_selector | stage65_adapter | 8 | 19 | 0.4164548657442394 | 0.1700189917495376 | 0.2529745478379099 | 0.6862830127540388 |
| gap_endpoint_norms | energy_regression | mlp_selector | stage65_adapter | 16 | 18 | 0.3392114285379648 | 0.15503261362512907 | 0.2512796148657799 | 0.634711515572336 |
| gap_endpoint_norms | topk_bce | mlp_selector | linear | 4 | 23 | 0.33506333568821783 | 0.29271199392235797 | 0.6033855663693469 | 0.4690889767978502 |
| gap_endpoint_norms | topk_bce | mlp_selector | linear | 8 | 19 | 0.35582716998301056 | 0.3219893998221347 | 0.6419753777353387 | 0.4838355026747051 |
| gap_endpoint_norms | topk_bce | mlp_selector | linear | 16 | 18 | 0.25578766336871517 | 0.2475775974906153 | 0.5684697247213788 | 0.4097355744904942 |
| gap_endpoint_norms | topk_bce | mlp_selector | stage65_adapter | 4 | 23 | 0.40695699142373126 | 0.1606864307237708 | 0.23996665620285532 | 0.6832091108612393 |
| gap_endpoint_norms | topk_bce | mlp_selector | stage65_adapter | 8 | 19 | 0.4181112019639266 | 0.17023401785837977 | 0.2529745478379099 | 0.6859801097920066 |
| gap_endpoint_norms | topk_bce | mlp_selector | stage65_adapter | 16 | 18 | 0.3303189207282331 | 0.1518372574614154 | 0.2512796148657799 | 0.6215511328644223 |
| gap_endpoint_rank | energy_regression | mlp_selector | linear | 4 | 23 | 0.3446884796671245 | 0.2929487195999726 | 0.6033855663693469 | 0.46928358855454816 |
| gap_endpoint_rank | energy_regression | mlp_selector | linear | 8 | 19 | 0.36839249651683004 | 0.3236427134589145 | 0.6419753777353387 | 0.4845588317042903 |
| gap_endpoint_rank | energy_regression | mlp_selector | linear | 16 | 18 | 0.2587568560718662 | 0.25036126747727394 | 0.5684697247213788 | 0.41143113126357395 |
| gap_endpoint_rank | energy_regression | mlp_selector | stage65_adapter | 4 | 23 | 0.40484559730343195 | 0.16103507837523584 | 0.23996665620285532 | 0.6845648729282877 |
| gap_endpoint_rank | energy_regression | mlp_selector | stage65_adapter | 8 | 19 | 0.41634063344252736 | 0.16972093244916514 | 0.2529745478379099 | 0.6850086732914573 |
| gap_endpoint_rank | energy_regression | mlp_selector | stage65_adapter | 16 | 18 | 0.334825467525257 | 0.1537107245789634 | 0.2512796148657799 | 0.6292625152402453 |
| gap_endpoint_rank | topk_bce | mlp_selector | linear | 4 | 23 | 0.33533463789069134 | 0.2913952949254409 | 0.6033855663693469 | 0.4669716241567031 |
| gap_endpoint_rank | topk_bce | mlp_selector | linear | 8 | 19 | 0.35714081870882136 | 0.3215885374106859 | 0.6419753777353387 | 0.4822451535024141 |
| gap_endpoint_rank | topk_bce | mlp_selector | linear | 16 | 18 | 0.2587870030353467 | 0.24927443265914917 | 0.5684697247213788 | 0.411947766939799 |
| gap_endpoint_rank | topk_bce | mlp_selector | stage65_adapter | 4 | 23 | 0.40928070765474567 | 0.16046723658623901 | 0.23996665620285532 | 0.682865233524986 |
| gap_endpoint_rank | topk_bce | mlp_selector | stage65_adapter | 8 | 19 | 0.4203244165370339 | 0.17039411985560468 | 0.2529745478379099 | 0.6872211286896154 |
| gap_endpoint_rank | topk_bce | mlp_selector | stage65_adapter | 16 | 18 | 0.33110267255041337 | 0.15231002908613947 | 0.2512796148657799 | 0.623305180006557 |
| stage100_base | energy_regression | mlp_selector | linear | 4 | 23 | 0.3431432687717935 | 0.2935174204733061 | 0.6033855663693469 | 0.47064504675243213 |
| stage100_base | energy_regression | mlp_selector | linear | 8 | 19 | 0.3649798710095255 | 0.3232088677192989 | 0.6419753777353387 | 0.4840805326637469 |
| stage100_base | energy_regression | mlp_selector | linear | 16 | 18 | 0.25720443747316796 | 0.24980945057339138 | 0.5684697247213788 | 0.410425061153041 |
| stage100_base | energy_regression | mlp_selector | stage65_adapter | 4 | 23 | 0.40308806429738586 | 0.16121713363605997 | 0.23996665620285532 | 0.6851413677568021 |
| stage100_base | energy_regression | mlp_selector | stage65_adapter | 8 | 19 | 0.4134420438816673 | 0.16959427689251147 | 0.2529745478379099 | 0.6845970812596773 |
| stage100_base | energy_regression | mlp_selector | stage65_adapter | 16 | 18 | 0.3367697604828411 | 0.154903387857808 | 0.2512796148657799 | 0.6341007451216379 |
| stage100_base | topk_bce | mlp_selector | linear | 4 | 23 | 0.32920097268146015 | 0.29305959849253943 | 0.6033855663693469 | 0.4703672748544942 |
| stage100_base | topk_bce | mlp_selector | linear | 8 | 19 | 0.35064397046440526 | 0.32177531876062093 | 0.6419753777353387 | 0.4822907981119658 |
| stage100_base | topk_bce | mlp_selector | linear | 16 | 18 | 0.2579128210329347 | 0.2517792292767101 | 0.5684697247213788 | 0.41649167074097526 |
| stage100_base | topk_bce | mlp_selector | stage65_adapter | 4 | 23 | 0.4132322079461554 | 0.16204810336880063 | 0.23996665620285532 | 0.6889743960422018 |
| stage100_base | topk_bce | mlp_selector | stage65_adapter | 8 | 19 | 0.42220921579160187 | 0.17060783230944684 | 0.2529745478379099 | 0.6881267146060341 |
| stage100_base | topk_bce | mlp_selector | stage65_adapter | 16 | 18 | 0.33598601383467513 | 0.1541641718811459 | 0.2512796148657799 | 0.6302359286281798 |

## Best Learned Feature By Group

| base | gap | feature mode | objective | energy recall | relative recall | precision@keep |
|---|---:|---|---|---:|---:|---:|
| linear | 4 | stage100_base | energy_regression | 0.2935174204733061 | 0.47064504675243213 | 0.3431432687717935 |
| linear | 8 | gap_endpoint_norms | energy_regression | 0.32395490298145696 | 0.4845810360030124 | 0.36720735462088333 |
| linear | 16 | stage100_base | topk_bce | 0.2517792292767101 | 0.41649167074097526 | 0.2579128210329347 |
| stage65_adapter | 4 | stage100_base | topk_bce | 0.16204810336880063 | 0.6889743960422018 | 0.4132322079461554 |
| stage65_adapter | 8 | stage100_base | topk_bce | 0.17060783230944684 | 0.6881267146060341 | 0.42220921579160187 |
| stage65_adapter | 16 | gap_endpoint_norms | energy_regression | 0.15503261362512907 | 0.634711515572336 | 0.3392114285379648 |

## Notes

- Extra features are decoder-available only: reference gap, endpoint motion norms, and endpoint rank.
- Target dense anchors are used only for offline labels/metrics.
- This stage tests feature quality before any residual-value prediction or rendered validation.
