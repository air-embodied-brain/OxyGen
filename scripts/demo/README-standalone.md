# 亮色横版 demo 视频（1080p）

入口：`scripts/demo/render_standalone.py`。2026-10-13 经用户单帧审阅确认的 v2 布局。
与 `render_comparison.py`（旧版合成器）并存，不替换网页播放器。

## 运行环境

在 jinfu-4090-1 上使用 `/home/lixiangyu/miniconda3/bin/python`（已安装 Pillow，FreeType 可读取 WOFF2），以及 PATH 中的 ffmpeg/ffprobe。
脚本只依赖 Python 标准库和 Pillow。需先安装网站 npm 依赖，供 Source Serif 4 字体读取；另外两种字体位于 public/fonts。

输入随网站仓库定位，无用户目录硬编码：
- `public/videos/*-interactive.mp4`：三个任务同步回放视频，1792×650、30 fps。
- `public/data/demo-timelines.json`：阶段、调用、完成输出和步数。
- `src/data/site.ts`：任务名称与指令。
- `public/fonts/`、`node_modules/@fontsource-variable/source-serif-4/`：字体。

## 单帧与完整视频

```sh
cd /home/lixiangyu/oxygen_ws/OxyGen-website
PYTHON=/home/lixiangyu/miniconda3/bin/python
$PYTHON scripts/demo/render_standalone.py --mode stills --task all --output ../review-artifacts/demo-video-1080p-v2-20261013/previews
$PYTHON scripts/demo/render_standalone.py --mode video --task all --output ../review-artifacts/demo-video-1080p-v2-20261013/videos
```

默认 mode 为 stills；output 必须显式传入。指定单任务可用 `--task moka-pots`，单帧时间可加 `--time 27.0`。另外两个 slug 为 `white-mug-and-chocolate-pudding`、`alphabet-soup-and-tomato-sauce`。
完整视频编码为 H.264、1920×1080、30 fps、CRF 18、slow、yuv420p、faststart、无声；保持 1×速度，结尾停留 2 秒。拒绝覆盖已有 MP4；中断可能留下不完整文件，确认后删除或换输出目录重试。

耗时任务建议放 tmux：
```sh
tmux new-session -s oxygen-demo-render
# 执行上述 video 命令；Ctrl-b d 脱离，tmux attach -t oxygen-demo-render 返回。
```

## 布局和时序

`compose()` 集中管理布局。标题 42 px、方法名 38 px、输出 34 px、状态 30–32 px；两路仿真各 640×479，等比例缩放；输出最多两行，超出会主动报错。无按钮、展开窗口或拖动提示。
`state()` 仅显示 `memory.time <= t` 的最新已完成文本。网站已知提前显示问题见 `KNOWN-ISSUES.md`。

时间轴采用网站数据：记录的推理总时间与 20 Hz 执行时间；推理内部阶段由 timing ratios 重建，非独立实测时间戳。推理累计值不含执行时间。素材来源、哈希验证见原始交付目录下的 STATUS.md / validation.json。

最终交付目录：`/home/lixiangyu/oxygen_ws/review-artifacts/demo-video-1080p-v2-20261013/`。
生成视频、预览、日志应置于已忽略的 review-artifacts，不提交 git。脚本及本说明是可版本管理的源资产。
