# 中澳PM2.5条件迁移：导师审阅与复现说明

版本：2026-09-13。课程COMP5702，作者暂定Sheen Wang。

## 阅读入口

- `manuscript/main.pdf`：英文研究报告，包含扩展方法、解释、局限与答辩准备。
- `manuscript/main_zh.pdf`：按英文内容章节对齐的完整中文理解版，不是仅翻译摘要，也不冒充逐句对译。
- `manuscript/main_zh.html`：单文件中文备用阅读版，图像内嵌；PDF预览无法加载时，用浏览器直接打开。
- `manuscript/journal.pdf`：更紧凑的期刊改写稿，尚未提交、尚未指定最终期刊模板。
- `REVIEW_CHANGES_ZH.md`：36条修改前后对照、原因、所在章节与文献网址。
- `AUTHOR_CHECKLIST_ZH.md`：需要作者和导师最终核对的事项。

已完成的分析不等于已证实正向迁移：新增监测目标90组配置中26组通过初步发现筛选，0组同时通过完整时间和新城市审计。前一卫星产品分支为0/96。两批配置共享历史和研究设计，不是186个独立研究。

## Overleaf编译

上传干净的 `overleaf_review_20260913.zip`，项目设置选择 **XeLaTeX**。

主文件有三个，按阅读目标选择：

| 主文件 | 内容 |
| --- | --- |
| `main.tex` | 英文完整报告 |
| `main_zh.tex` | 中文理解版 |
| `journal.tex` | 期刊改写稿 |

参考文献使用BibLaTeX APA样式与Biber；Overleaf正常重编译流程会处理。中文使用TeX Live附带的Fandol字体，不需要上传Windows字体。

此前截图中的 `CTeX fontset fandol is unavailable in current mode` 和 `pdflatex ... unihei51 not found` 来自中文稿使用了不合适的pdfLaTeX编译模式。切换XeLaTeX后从头编译；不要继续沿用原项目的 `main.aux`、`main.bbl`、`main.fdb_latexmk` 等缓存。新压缩包不包含这些文件。

标题、作者、课程与完整参考文献均包含在包内。没有补写未经确认的SID、导师、基金、利益冲突或正式提交声明。

## 文件层次

`data`保存对齐监测面板、站点月、城市区边界和共同ERA5月摘要。`results`按队列/变量组/方向划分，每种组合包含3个城市划分与3个标签窗口。`audit`保存来源、质量、对照指标与核验。

数值以 `audit/all_metrics.csv`、`audit/all_gates.csv`、`audit/all_budgets.csv`、`audit/frozen_candidates.csv` 和 `audit/replication.csv` 为最终聚合表。早期运行中的 `monitor_all_*` 不作为本版最终汇总。

模型结果定位需要同时使用：队列、变量、方向、种子、窗口、阶段、覆盖选择和模型。不能把不同目标、不同时间或不同接受行的“最好分数”拼成一个模型成绩。

## 三个复现层级

### 1. 只重建论文

干净Overleaf包即可编译三份PDF，不需要Python、原始数据、API或模型。三个HTML文件也不依赖在线图像。

### 2. 核验保存结果

在保留原有目录结构的研究包中，安装 `environment_versions.json` 列出的科学计算依赖，然后运行：

```powershell
python -m unittest test_invariants
python audit_results.py
```

应得到6项测试通过、90组配置核验通过。默认CSV解析可能改变阈值边界小数；复读保存预测时必须使用 `pd.read_csv(path, float_precision='round_trip')`。只有4条边界受到默认解析的影响，原始模型和锁定条件未修改。

### 3. 从分析面板重新训练

不要覆盖交付的 `results`。将代码、`data`、`audit/build_summary.json` 和冻结协议放入一个新的研究目录，且保持 `common.py` 所需的同级共同月度目录关系；新目录不复制任何 `results/*/manifest.json`。

执行主实验与两个扩展：

```powershell
python run_monitor_models.py
python run_supplement_models.py --experiment outer_time
python run_supplement_models.py --experiment mixing
python audit_results.py
```

脚本对已存在的完成清单会跳过，以保护结果。完整拟合会花费明显时间；仅重复文稿编译无需重训。模型库或线程环境变化可能导致微小数值差异，版本清单保留实测环境。

`source_models.joblib`只保存本研究的源模型与变量定义；目标滚动拟合由脚本重新建立。只读取本项目可信的joblib，未知来源的序列化文件可能执行代码。

### 原始档案重建的额外条件

完整小时档案和最初卫星/辅助面板在原项目目录，不全部复制进紧凑交付包。原路径可由环境变量 `PM25_ORIGINAL_ROOT` 指定；来源清单列出了依赖路径与哈希。上一共同月度分支作为同级 `common_monthly_transfer_20260912` 保留，制图读取其 `modelling_cache.joblib`。

3.52GB ERA5原始压缩块保留在本机 `data/raw_weatherbench2`，紧凑包仅含149,664行摘要、下载程序和完整块级清单。`collect_common_pbl.py`可从公开WeatherBench2源重新获取；不能因压缩包不附大体积原块而宣称原始数据全部内嵌。

首次小时重建会记录质量计数。维州现有按年缓存重用时不会重新生成所有原始小时质量计数；若从原始档案重建，应在全新的研究目录执行，避免覆盖已有审计记录。当前报告的计数来自已保留的首次完整运行。

重新从Markdown生成文稿需要Pandoc 3.11，并将其可执行程序放到 `tools/pandoc-3.11/pandoc.exe`；该官方工具不重复打入研究压缩包。`build_documents.py`重新生成图表、双语结果段与TeX/HTML，`compile_documents.py`调用本机XeLaTeX/Biber。PDF页面检查另外使用附带pdfplumber、pypdf、Pillow和Poppler的文档环境；这些不属于模型训练依赖。直接使用干净Overleaf包无需上述本地构建环境。

## 协议与质量边界

`protocol_observed_frozen.md`是主模型拟合时使用的STATUS内容快照；后续STATUS仅更新完成情况。`EXTENSION_PROTOCOL.md`记录部分主结果查看之后新增的探索扩展。原始构建阶段和拟合阶段的协议哈希可不同，不把后补说明冒充同一份预注册。

本版核验范围：保存预测的指标回算、逐行规则重现、城市划分、训练时间、核函数基本性质与自助法小样本对照。并未独立重写全部模型，也不保证机构原始仪器读数没有误差。

尚存限制：澳洲只有三个州的当前交集；NSW缺小时完整度；中澳负值处理不完全相同；昆州归档时区不确定；站点搬迁/仪器历史不完整；ERA5格网粗；城市与气象网格之间仍可能有相关性；所有新实验仍是历史探索。它们已进入正文，不藏在交付说明里。

## 对外使用

这是本机导师审阅包，未上传公共仓库、未投稿、未访问Polymarket账号。模型、指标、图表及文稿可本地查看；在公开发布原始或派生数据前仍应核对各来源许可。AI披露不能代替Canvas允许范围，作者须完成清单后决定提交。
