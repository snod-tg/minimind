# 简历与 AI Infra 面试资料

## 常用入口

- [CommLab：C++ RC verbs 通信引擎与变长 All-to-All](comm_lab/README.md)：独立新项目，以C++ RC verbs引擎为主线，含起步代码、六阶段路线与验收。

- [InfiniBand / RDMA / NCCL / HCCL 快速教程](communication_tutorial/README.md)：67 篇短文、4 张图示、阅读路线、实践命令和官方资料来源，每篇正文不超过 1000 个字符。
- [算子研究 · 主题阅读页](research_kernel/site/index.html)：离线阅读、侧栏导航、全文搜索、文章目录、明暗主题。
- [算子研究 · Markdown 索引](research_kernel/README.md)
- [简历 v16 源码](resumes/tongleiwen_ai_infra_resume_20261006_v16.tex) · [PDF](output/pdf/tongleiwen_ai_infra_resume_20261006_v16.pdf)
- [v16-style1](output/pdf/tongleiwen_ai_infra_resume_20261006_v16-style1.pdf)：参考 `cv.pdf`，黑白衬线字体、左对齐页首与右侧照片。
- [v16-style2](output/pdf/tongleiwen_ai_infra_resume_20261006_v16-style2.pdf)：参考黄一航简历，蓝色标题、居中无照片页首与无衬线字体。
- [v16-style3](output/pdf/tongleiwen_ai_infra_resume_20261006_v16-style3.pdf)：参考董兴震简历，中文楷体、英文 Times New Roman、图标标题、学校标签与浅灰项目标题条。
- [面试题 v2](interview/questions/简历面试提问_v2.md) · [对应答案](interview/answers/answer-v2.md)
- [项目难点面试准备](interview/questions/meetquestion.md)

## 文件分类

| 文件夹 | 内容 |
| --- | --- |
| `resumes/` | 各版本简历 TeX、模板与简历文案，版本名完整保留 |
| `output/pdf/` | 已生成的简历 PDF |
| `interview/questions/` | 分类题单、v-1/v2 与项目难点题 |
| `interview/answers/` | 原始回答、修订版和标准答案 |
| `interview/handwritten/` | 手写答案照片 |
| `assets/` | 证件照、校徽等简历素材 |
| `references/` | 参考简历 PDF |
| `plans/` | 项目学习计划、实验待办 |
| `docs/` | 工作要求、测试命令、归档记录 |
| `research_kernel/` | 14 类算子的 Markdown 研究资料及主题阅读站 |
| `comm_lab/` | 独立通信项目的设计、起步代码、命令、测试与结果状态 |
| `communication_tutorial/` | InfiniBand、RDMA、RoCE、NCCL、HCCL 的短篇教程、命令与面试问答 |
| `result-datas/` | 实验数据、证据说明与复现脚本 |
| `benchmarks/` | 专项 benchmark |
| `tools/` | 阅读页构建工具 |
| `tmp/` | 历史临时材料，保留现状 |

## 主题阅读

直接用浏览器打开 `research_kernel/site/index.html`，不需要启动服务或联网。所有正文、目录和搜索内容都由原 Markdown 生成，数据来源链接保持可点击。

修改 Markdown 后，需要重新生成页面；方法见 [阅读主题说明](docs/kernel_reader.md)。在 VS Code 中，工作区还提供 Markdown 预览样式；其他编辑器的原生预览是否支持主题，取决于编辑器本身。

## 归档记录

本次移动了根目录的 36 个文件，没有删除资料。原路径、新路径和校验值保存在 [归档清单](docs/archive_manifest_20260929.json)。Markdown 链接和简历图片搜索路径已随目录调整。历史实验原始数据与已有 PDF 未移动。

TeX 源文件可从 `resumes/` 或仓库根目录编译，图片通过 `assets/` 的相对搜索路径查找。
