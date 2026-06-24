# Questions And Decisions Log

## 2026-06-25

### User Question

放在这个 tmp 文件夹下没啥问题吗？希望放到 `/mnt/hdd2tC/haocheng` 下面。另外，这个代码后续是否应该推送到 NeoVerse 仓库，还是重新建一个仓库，希望指导新建。

### Answer / Decision

建议不要长期放在 `/mnt/hdd2tC/tmp/opencode`，该目录更适合作为临时实验区。长期项目放到 `/mnt/hdd2tC/haocheng/KeyStreamSplat` 更清晰。

建议不要推送到 NeoVerse 仓库。NeoVerse 当前已有 CompactWorld/CWGS 相关改动，且 KeyStreamSplat 的研究目标、rate accounting 和代码结构会逐渐独立。应新建独立仓库。
