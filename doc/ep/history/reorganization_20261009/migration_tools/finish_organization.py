from pathlib import Path
import ast
import json
import re
import shutil

K=Path('/Users/tongleiwen/Code/my_cuda_kernels');M=Path('/Users/tongleiwen/Code/minimind')
W=Path('/Users/tongleiwen/Documents/resume')
verification=json.loads((W/'organization_work/verification.json').read_text())
results={K:('3','通用包边界、原生库解析、三种构建的正式源码路径和 EP --fmad=false'),
         M:('5','模型模块导入、KN 权重往返与共享参数、缓存失效、专家数/Top-K/GPU配置、tokenizer与训练参数路径')}
for root,(count,scope) in results.items():
    doc=root/'doc' if root==K else root/'doc/ep'
    path=doc/'results/organization_20261009.md'
    assert not path.exists()
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(f'''# 本地项目整理验收（2026-10-09）

## 目的
把通用 CUDA 算子和 MiniMind 模型代码按职责归位，消除正式执行入口对阶段目录和跨项目 sys.path 的依赖。

## 执行方案
先拆分通用包与模型适配，随后统一编译/训练入口，迁移文档与图表，最后验证导入、数值实现和文档链接。具体命令见 [操作指令](../commands/local_workflow.md)。

## 结果
- 本项目 {count} 项 CPU 单元检查全部通过：{scope}；两个项目合计8项。
- ep、grouped_gemm、kernel9 三种编译命令预览通过，读取唯一 dev/cuda 源码；6选2/tmux 训练命令预览通过。
- CUDA 的17个源文件哈希在目录迁移前后相同；12个前向/工作区/模型适配定义与原件比较未改变计算逻辑，原 BF16 兼容实现逐字节相同。
- 正式 Python 入口不再包含服务器阶段路径或跨项目 sys.path；共享 benchmark 参考实现已统一。
- 说明、命令、结果、图表和历史实验按项目归位；两个项目的当前文档入口相互链接。
- 使用已有 py313 环境完成 editable 安装；本轮不启动训练、不改服务器配置、不改 CUDA 计算策略。

本地没有 CUDA GPU，本轮未执行 CUDA 编译、Compute Sanitizer、多卡训练或性能复测；此前 GPU 数值和性能结论仍属于其原实验轮次。训练模型质量也未在本轮验证。

## 调整闭环
检查中发现旧入口依赖阶段目录、参考文件重复、tokenizer预检路径不正确，已分别改为普通包导入、统一参考实现和模型根目录 tokenizer。修正后重跑对应 CPU 检查通过。后续部署到 CUDA 环境时先重建库，再运行单算子与模型回归，完成真实文本短训练和恢复验证后才启动新的独立训练。
''')
    index=doc/'README.md'
    assert not index.exists()
    rows=[]
    for file in sorted((doc/'design').glob('*.md')):
        if file.name=='project_structure.md':continue
        name=file.name
        cells=[f'[{kind}](./{kind}/{name})' if (doc/kind/name).exists() else '—' for kind in ['design','commands','results']]
        rows.append('| '+file.stem+' | '+' | '.join(cells)+' |')
    index.write_text('''# 项目文档

[职责与目录](design/project_structure.md) · [当前操作指令](commands/local_workflow.md) · [整理验收](results/organization_20261009.md) · [环境](environment.md)

以下阶段文档记录对应历史实验，当前执行入口以项目 README 和操作指令为准。

| 阶段 | 说明 | 指令 | 结果 |
| --- | --- | --- | --- |
'''+ '\n'.join(rows)+'\n')

issues=[];checked=0
for root in (K,M):
    doc=root/'doc' if root==K else root/'doc/ep'
    pages=list((root/'analysis').glob('*.md'))+[root/('readme.md' if root==K else 'readme_ep.md'),doc/'README.md']
    for part in ['design','commands','results']:pages+=list((doc/part).glob('*.md'))
    for path in pages:
        text=re.sub(r'```.*?```','',path.read_text(),flags=re.S)
        text=re.sub(r'`[^`]*`','',text)
        for target in re.findall(r'!?\[[^\]]*\]\(([^\s)]+)\)',text):
            if '://' in target or target.startswith(('#','mailto:')):continue
            target=target.split('#')[0];checked+=1
            if target and not (path.parent/target).exists():issues.append((str(path),target))
assert not issues,issues
syntax=0
for root,folders in [(K,['dev/pytorch/my_cuda_kernels','script','tests']),
                     (M,['model/ep','model/backends','scripts/ep','tests/ep'])]:
    for folder in folders:
        for path in (root/folder).rglob('*.py'):ast.parse(path.read_text());syntax+=1
verification.update(cpu_tests_passed=8,markdown_links_checked=checked,broken_links=0,active_python_files_parsed=syntax)
(M/'doc/ep/results/organization_20261009.json').write_text(json.dumps(verification,indent=2)+'\n')
(K/'doc/results/organization_20261009.json').write_text(json.dumps(verification,indent=2)+'\n')
tools=M/'doc/ep/history/reorganization_20261009/migration_tools'
assert not tools.exists()
shutil.move(W/'organization_work',tools)
for name in ['docs','commands','results']:
    path=W/'minimind_ep_grouped_gemm'/name
    if path.exists() and not any(path.iterdir()):path.rmdir()
print(json.dumps(verification,indent=2))
print('LOCAL ORGANIZATION COMPLETE')
