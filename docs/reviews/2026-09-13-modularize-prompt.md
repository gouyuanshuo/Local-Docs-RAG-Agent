# Prompt：先闭合测量接缝，不要拆已经深的模块

把下面整份原样交给编码 agent。不要再追加「顺便拆 qdrant_store / 上 cross-encoder / 做历史上传」。

---

你是这个仓库的实现 agent。仓库根：`D:\Playground\Local Docs RAG Agent`。分支 `GYS`，审计时 HEAD `15b7363`（相对 `origin/GYS` ahead 1）。先核对 `git status` 与 `git rev-parse HEAD`。工作区当时干净。不要改 `tasks.md` 以外的任务板叙事到「Phase C 已完成可指导迭代」。不要开 PR、不要 push，除非我另外授权。

## 一句话结论（交回第一行必须重复）

工程骨架已经强过早期原型；**测量闭环未合上**。官方优先级是 eval 加固，不是再分层。本轮「模块化」= 把 **Eval / Compare / Gold / Ingest 生命周期** 收成可独立测试的接缝。禁止把 AppConfig、reporting、qdrant_store、CacheManager、cross-encoder、query rewrite、历史上传、配置面板、Ollama、Docker 当本轮工作。

审计报告：`docs/audit-report-2026-09-13.html`。质量门当时全绿（ruff / mypy / pytest 144 passed 84% / `pnpm run build`）。未跑 live Qdrant / 真模型 / 浏览器点选。

## 目标

产品链是 `Agent -> Tools -> RAG -> Citation -> Eval`。当前断在 Eval：

- 金标有一条永远不是原文连续子串 → Q1 的 span/RR 上限锁在 2/3（F-001）
- compare 把 chat/embedding/runtime fallback 记成 `status=ok` 并进 leaderboard（F-002）
- 两问 + 一篇讲义 + 默认 `DOCS_DIR=docs` 无法分开四种检索策略（F-003）
- Qdrant delete-then-upsert 失败后，增量 ingest 因 checksum 匹配而永不自愈（F-004）
- Qdrant 上 `blended ≡ dense`，和 local blended 不是同一排序（F-005）

先让测量能分开策略、能拒绝降级、能抓住 ingest 窗口。然后再谈拆文件。

项目原则（`AGENTS.md`）：**可测量 > 可解释 > 可靠 > 更像 agent > 更像产品。**  
硬规则：不要把 fallback 当成功；选项集只在 `constants.py` 声明一次；包外只从 `local_docs_rag_agent.rag` 导入。

## 先读

1. `AGENTS.md`、`spec.md`、`tasks.md`、`docs/development-roadmap.md`
2. `skills/backend-api/SKILL.md`、`skills/review-bugfix/SKILL.md`
3. `docs/audit-report-2026-09-13.html`（或同内容的审计正文）
4. `evals/harness.py`、`evals/comparison.py`、`data/evals/sample_eval.jsonl`
5. `rag/ingest.py`、`rag/qdrant_store.py`、`rag/pipeline.py`、`rag/retrieval.py`
6. `runtime/{dispatch,basic,agents_sdk}.py`、`tools.py`、`providers/chat.py`、`providers/embedding.py`

## 非目标（硬停止）

- 拆 `AppConfig`、抽 reporting 模块、拆 `qdrant_store.py`、加 `CacheManager` / `StoreService` / `RankingBackend` / `ToolRegistry`
- 为「语义对称」向 Qdrant 要向量；为 F-001 改 RR 公式或放宽成 token 集合匹配
- Ask 路径 fallback 改成 503（Ask 200 + 诊断芯片是对的）
- cross-encoder、query rewrite、弱检索重试、tool trace、Phase E runtime 大改
- 前端历史上传、配置面板、React Router、ESLint 大工程、Docker、Ollama
- 为覆盖率 mock 整张 ingest 矩阵；给 `pipeline.py` 再开只 assert 调用顺序的浅测试
- 重写整份 roadmap；把 Phase C 写成「两端质量已可比」
- 未授权时跑会写入远程 collection 的 live Qdrant / 真模型脚本

## 本轮接缝（只动这些）

| 接缝 | 拥有 | 不拥有 | 本轮必须变成什么 |
| --- | --- | --- | --- |
| **Gold** | `sample_eval.jsonl` + 「关键字必须是 source 连续子串」测试 | 指标公式 | F-001 修好；Q1 不再被 2/3 锁死 |
| **EvalCorpus** | eval 用的讲义/语料路径；与 `docs/` 工程文档分离 | chunk 策略实现 | `DOCS_DIR` 指向 sample 或 exclude 架构文件；8–15 条按失败模式写的多文档 case |
| **CompareAggregation** | 格子 status、leaderboard 准入、排序主键 RR | Ask HTTP 状态码 | fallback 或 `requested_runtime != actual_runtime` → `degraded`/`error`，不进榜；有排序测试 |
| **RetrievalSemantics** | 文档与 compare 默认轴 | 新抽象层 | 写明 Qdrant blended≡dense；比 backend 时扫 `dense` 与 `hybrid_rrf`，不要扫 blended |
| **IngestLifecycle** | delete-then-upsert 失败后的 dirty/needs_reindex | 分布式事务、换向量库 | checksum 匹配不得把「Qdrant 里已删掉的源」当成已索引；假 client 打到 `QdrantChunkStore` 分支，不是只打 Fake 鸭子类型 |
| **EvalVisibility** | Compare UI 能看见降级 | 类型生成、历史上传 | 至少把 `CompareRun.summary` / fallback 警告渲染出来；不要先做 OpenAPI 类型生成 |

可顺手、不得单开大 PR 膨胀：

- `frontend` `tsc -b --noEmit` 或 node tsconfig `noEmit`（F-009）
- SDK 工具改走 `tools.search_documents` 并返回 status（F-012）；不要新建 ToolRegistry
- Ask `max_length`（例如 4000）
- 文档质量门命令与 CI 对齐；Phase C 改成「代码完成，测量未闭合」（F-010）

保留且不要再包一层：`pipeline.retrieve` 单入口、`presenters`、AXES 派生、`shared._render_hit_block`、`LocalDocsAgent` 门面、`file_io` 原子写、citation `[S1]` 从第 0 列开始。

## 工作方式（DSH 合同，不是拆包）

1. **先盘点再动刀。** 列出每个接缝的文件、现有测试、将要加的会红测试。不要先搬 `qdrant_store`。
2. **一次一个发现 ID。** 提交信息：`fix(eval): gold keyword substring`、`fix(compare): fallback not ok`、`fix(ingest): dirty source after upsert fail`。禁止「全面模块化」。
3. **先写会红的测试。**  
   - Gold：每条 `expected_retrieval_keywords` 必须是对应 source 文件的连续子串。  
   - Compare：调用真正的矩阵/排序路径（`_run_matrix_case` / `_build_leaderboard`），断言 fallback 不进榜、RR 高的行在前。把 sort key 改回 `retrieval_span_hit_rate` 必须红。  
   - Ingest：假 client 上 delete 成功、upsert 失败；下一次相同 checksum 必须再写回或标 dirty。`isinstance(..., QdrantChunkStore)` 分支必须被打到。
4. **不要把 fallback 当成功。** 这是 `AGENTS.md` 硬规则。live 脚本已经拒绝降级；harness/compare 必须对齐。
5. **窄证据。** 每块跑相关 pytest 文件。终检跑审计用过的门：
   - `python -m ruff check backend/src tests scripts`
   - `python -m ruff format --check backend/src tests scripts`
   - `python -m mypy`（裸 mypy，含 tests/scripts，与 CI 一致，不要只跑 `backend/src`）
   - `python -m pytest --cov=local_docs_rag_agent --cov-report=term-missing`
   - 若动前端：`pnpm run build`
   覆盖率不得无故下降。未跑 live / 浏览器写未跑。
6. **更新 `tasks.md` 与 roadmap 状态句。** Phase C = 代码完成、测量未闭合。不要声称 local vs qdrant blended 质量可比。
7. **独立 worktree。** 不要在别人的脏树上叠。

## 建议提交顺序（与审计 §9 一致）

1. F-001 金标关键字 + 子串测试  
2. F-003 扩 eval 集 + 语料与工程文档分开（依赖 1）  
3. F-002 + F-006：fallback 不进榜 + leaderboard 排序测试  
4. 在能区分的集合上跑 eval-compare，把谁赢写进交接（可用 hash embedding；不要把 fallback 当赢家）  
5. F-004 ingest dirty + 假 client 测 delete-then-fail  
6. F-005/F-010 文档：blended≡dense、质量门命令、Phase C 措辞  
7. 顺手 F-009 / F-012 / Ask max_length / Compare 降级可见  
8. **停。** JSONL 缓存、拆 qdrant_store、Phase E–H 等第 4 步证明策略能被样本分开之后再做。

## 完成时交回

1. 第一行：测量闭环做到哪一步；未声称可实盘级 RAG 质量。
2. 每个 F-00x 的状态：FIXED_VERIFIED（测试名）/ 文档 / 未做。
3. 新 eval 集能否在至少两种 retrieval 策略上打出不同 RR；贴命令与数字。若仍打平，说明是样本还是实现。
4. compare 在无 key / hash fallback 时 leaderboard 为空或全是 degraded，而不是假赢家。
5. ingest 失败窗口的测试：修复前红、修复后绿。
6. 质量门命令与退出码。live Qdrant / 浏览器未跑。
7. 明确下一叠不要做的事。

## 验收

- 金标里不再存在「全文找不到的连续子串」。
- 改 leaderboard 排序主键会使测试红。
- 改 compare 把 fallback 标回 ok 会使测试红。
- 假 Qdrant：delete 后 upsert 失败，再 ingest 同一文件必须试图恢复或拒绝「已索引」声称。
- 没有新的上帝抽象。`pipeline.py` / AXES / presenters 仍在。
- citation 第 0 列契约未回潮。
- 没有人能从本轮 diff 里读出「可以开始做 Docker / Ollama / 历史上传」。
