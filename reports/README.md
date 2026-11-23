# 姓名掩码后的实体偏移重映射

本目录是 UID 73 的验收产物。能力入口是 `spacy.training.rewrite`，命令行入口是
`spacy rewrite-ents`。

## 问题

合同文本中的客户姓名被掩码替换为占位符后字符长度变化，随后所有命名实体的字符
位置整体偏移。旧的导入路径直接按原始偏移构造 `Span`，越界或错位的标注被静默丢弃。

## 接口

```python
from spacy.lang.en import English
from spacy.training import rewrite_annotations

nlp = English()
result = rewrite_annotations(
    "The contract was signed by John Smith on 2024-01-01.",
    [(27, 37, "[CLIENT]")],
    [(27, 37, "PERSON")],
    crossing="shrink",      # 跨界策略：shrink | drop
    conflicts="raise",      # 冲突策略：raise | keep_first | keep_longest
    alignment="contract",   # 边界对齐：none | strict | contract | expand
    on_error="raise",       # 清单校验失败：raise | skip
    tokenizer=nlp,
)
result.text_after        # 改写后的文本
result.to_table()        # 每条实体的原起止、新起止、处置结果
result.summary()         # 字符数 / 实体数 / 丢弃数 / 收缩数 对账表
```

- 偏移只由改写清单的偏移账本推导（`_Ledger.map_offset`），从不在新文本里搜索旧
  子串。因此当替换串本身包含原片段（例如 `"John Smith (John Smith)"`）时映射仍然
  正确。
- 词边界对齐复用 `Doc.char_span(..., alignment_mode=...)` 与 tokenizer 的切分结果，
  不另造边界规则。
- 处置结果五态：`kept` / `replaced`（完整落在替换片段内）/ `shrunk`（跨界收缩）/
  `adjusted`（为落到合法词边界而调整）/ `dropped`（丢弃，带 `reason`）。被丢弃的
  实体 `new_start`/`new_end` 固定为 `-1`，绝不会产生零长度实体。
- `RewriteResult.reconciles` 校验：`实体数(前) == 实体数(后) + 丢弃数`、
  `字符数(后) == 字符数(前) + 声明增量`。

## 语料导入

关闭时行为逐条不变。`Corpus` 新增可选 `rewriter` 钩子，默认 `None`：

```python
from spacy.training import Corpus, rewrite_doc
corpus = Corpus("train.spacy", rewriter=lambda doc, nlp: rewrite_doc(doc, ops, nlp=nlp).doc)
```

批量改写 DocBin（原子写入，失败不落半份文件，残留 `.partial` 会在下次运行时覆盖）：

```python
from spacy.training import rewrite_docbin_from_manifest
report = rewrite_docbin_from_manifest(
    "in.spacy", "out.spacy",
    {"0": [[27, 37, "[CLIENT]"]]},
    nlp=nlp,
)
```

输出沿用源 DocBin 的属性集合与版本号，旧产物仍可读、旧读取路径不受影响。

## 产物

| 文件 | 内容 |
| --- | --- |
| `rewrite_samples.json` | 七类样本的完整行级映射与对账表（整体替换、跨界收缩、跨界丢弃、相邻替换合并、增长+多处替换、空清单、清单校验跳过） |
| `determinism.json` | 同一输入连续执行 5 次，逐条输出一致 |
| `corpus_rewrite.json` | DocBin 批量改写的对账结果 |
| `corpus_documents.json` | 改写后 DocBin 读回的文本与实体 |
| `corpus/` | 改写前后的 `.spacy` 产物 |
| `existing_corpus_tests.json` | 原有语料读写测试的实际输出 |
| `new_rewrite_tests.json` | 新增测试的实际输出 |

## 复跑

```bash
PYTHONPATH=$PWD python reports/generate_report.py   # 生成 reports/
python -m pytest spacy/tests/training/test_rewrite.py -q
```

环境说明：本机缺少系统 gcc/g++，扩展使用 zig 的 `cc`/`c++` 作为编译器在
Python 3.13 虚拟环境中构建；生成脚本仅依赖仓库内代码，不引入额外依赖，可离线运行。
