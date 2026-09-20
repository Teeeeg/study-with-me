---
title: 科技大厂 AI 软件工程、财务与人才调研
description: 对照海内外科技大厂的 AI 软件开发实践、员工工作所用 AI 的预算与实际支出及人员变化，区分研究证据、企业自报和媒体报道。
lang: zh
ref: ai-enterprise-research
---

## 员工 AI 预算与实际支出


| 公司与金额时点 | 员工用途及覆盖范围 | 预算、可用额度或报销上限 | 实际花费：已知金额与缺口 |
| --- | --- | --- | --- |
| **腾讯**；2026-06-13报道 [33][34] | 主要为员工调用外部模型；由部门、组内分配，自研混元仍可不限量使用。 | 受访不同部门约**1,000–7,000元人民币／人／月**，其中包含外包岗位；混元团队约7,000元，优图约5,250元。公司总预算金额未给出。 | 有员工自述**1,400元额度两天耗尽**，另一位称**5,000元额度三天耗尽**。这是单人额度消耗，不是已核验的供应商结算；公司累计实付未取得。 |
| **阿里巴巴**；2026-06-13报道 [33] | 产研等 AI 使用较多的岗位，报道指可调用不同模型。 | 约**8,000元人民币／人／月**；是受访岗位额度，不是全员平均实支，集团员工工具预算总额未取得。 | **未取得公司或对应部门的实际累计费用、结算额。** |
| **字节跳动**；2026-06-13报道 [33] | 内部 TRAE 模型使用与额外外部模型报销分开。 | 报道称内部 TRAE 可不限量调用；**部分部门**外部模型按员工实际支出的**50%报销**，产研岗位上限**1,000美元／人／年**，其他岗位**300美元／人／年**。 | **未取得实际报销总额或内部 TRAE 成本。** 员工个人自费不计为公司实支，除非已报销。 |
| **Uber**；2026年4月及6月报道 [21][36] | 工程师使用 Claude Code 等编程工具；后续限额按工具单独计算。 | 6月报道上限为**1,500美元／人／工具／月**；原2026年度预算及追加预算的绝对总额未给出。 | 4月报道原年度预算已用尽，另提工程师 API 开销约**500–2,000美元／人／月**；没有代表性样本、完整账单或公司累计实付金额。 |
| **NVIDIA**；2026-03-20报道 [17][37] | 工程师使用 AI 智能体及推理资源；高管发言，不是采购结算披露。 | 黄仁勋以年薪50万美元工程师为例，期望其一年使用**至少25万美元**的 token；另提出额外配置约基本工资50%的 token 资源。**属目标／提议，未核验获批预算。** | **实际人均及团队结算额未取得。** 被问及工程团队20亿美元 token 开支时，他仅回答“正在努力达到”；未明确会计期间，不能写成已花20亿美元。 |
| **Microsoft**；2026-08-04报道 [35] | 员工内部 AI 工具，按事业部分配并监测用量。 | 报道称7月起各事业部有 token 预算上限；**具体上限和全公司预算金额未取得**。 | 报道提部分工程师每月 token 花费为**数百至数千美元**，但没有具体样本、结算账单或公司合计数。 |
| **Meta**；2026-06-17报道 [36] | 员工使用 Anthropic 等 AI 工具。 | 报道称收紧员工使用，**具体配额和获批总预算未取得**。 | **未取得可核验的已发生费用总额或实际付款。** |
| **Alphabet／Google**；截至2026-09-19检索 [10] | 已有 AI 编码采用数据，但采用率不是费用披露。 | **未取得员工 AI 工具专项预算总额或统一人均额度。** | **未取得专项实际费用或付款总额。** |
| **Amazon／AWS**；2026年4月至6月资料 [12][36] | 员工 Kiro 开发等；报道提及内部用量管理。 | **未取得员工工具专项预算或可比的人均上限。** | **未取得专项实际费用或付款总额。** |
| **Oracle**；截至2026-09-19检索 | 内部员工 AI 工作工具。 | **未取得可核验的专项预算或人均额度。** | **未取得专项实际费用或付款总额。** |
| **Salesforce**；技术资料2026-09-16 [20] | 内部员工流程中的 AI Agent；不等于对外客户用量。 | **未取得员工内部使用的专项预算或人均额度。** | **未取得专项实际费用或付款总额。** |
| **百度**；截至2026-09-19检索 [27] | 本次核验了 Comate 功能，未取得内部费用明细。 | **未取得员工工具专项预算或可比的人均额度。** | **未取得专项实际费用或付款总额。** |
| **华为**；截至2026-09-19检索 [29] | 本次核验了 CodeArts 产品能力，未取得内部费用明细。 | **未取得员工工具专项预算或可比的人均额度。** | **未取得专项实际费用或付款总额。** |

## 海外大厂对照


| 公司与资料时点 | AI 软件工程实践与效果 | 人员与组织变化 |
| --- | --- | --- |
| **Microsoft**；论文2025-06，人事费用资料2026-07-29 | 参与三家企业、4,867名开发者的随机试验；合并任务完成数增加26.08%。这是本表中较强的工程实证，但不是微软单独的全员提效数。[2] | 财报披露自愿退休计划费用及 Xbox 遣散、减值事项；该页未提供可用于计算 AI 工程师流失的分组数据。[9] |
| **Alphabet／Google**；工程2026-04-22，组织消息8月 | Pichai 称75%的新增代码由 AI 生成、工程师批准；一项复杂迁移较一年前人工方式快6倍。均为企业自报，任务、工具及流程也可能变化。[10] | 官方宣布 Jeff Dean、Sanjay Ghemawat 将成立独立公益公司（PBC），Google 为创始投资人及云伙伴；同时调整 DeepMind 领导职责。[11] |
| **Amazon／AWS**；2026年1月至4月资料 | 股东信称6名工程师借助 Kiro，76天完成 Mantle 推理引擎；“传统约40人一年”是管理层反事实估计，不是对照实验。另有 Kiro 相关局部故障报道，Amazon 对 AI 因果归因有异议。[12][14] | 1月28日官方宣布缩减约16,000个岗位，多数受影响美国员工有90天内部求职期。公告不能证明这些岗位由编码工具直接替代。[13] |
| **Meta**；技术2026-03-17，人事7月29日 | 广告排序研发智能体 REA 的首次部署：3名工程师交付8项模型改进上线提案，公司称相对历史人员配置达到5倍工程产出；关键决策仍有人监督。不是全公司或全部软件任务的测量。[15] | 6月30日员工75,472人，同比-1%；其中**仍包含约8,000名受5月裁员影响的员工**，多数到Q3末才退出人数统计。[16] |
| **NVIDIA**；2026-03-20高管发言 | 黄仁勋主张给予工程师额外 token 资源以使用智能体；未同时提供可比的净提效测量。[17][37] | token 预算发言不代表用 token 代替现金工资；本次未取得与内部 AI 编程直接对应的减员、离职率或实际预算发放人数。[17] |
| **Oracle**；人员2026年6月报道 | 本次未找到可比的内部 AI 编码、测试净提效测量。报道显示公司重组与 AI 采用同时推进，不能仅凭同期变化推断研发生产率。[18] | 路透依据年报：截至5月31日员工141,000人，上年162,000人，净减少约21,000人。披露的重组因素包括管理、产品、绩效、战略和并购。[18] |
| **Salesforce**；招聘2026-05-06，技术9月16日 | 自述通过模拟企业工作流、工具调用和强化学习开发 Koa；已在内部员工流程运行，客户仍为小范围试点。这是 Agent 软件研发与部署证据，**不是内部编码效率测量**。[20] | 宣布招聘1,000名 AI 原生毕业生和实习生，覆盖工程、产品、销售等岗位；是招聘承诺，不是已经入职的人数。[19] |
| **Uber**；2026-04-23媒体报道 | 报道转述 CTO：约95%工程师每月使用 AI，近70%的提交代码由 AI 生成。采用率、代码占比都不是质量校正后的交付效率。[21] | 报道当时仍在招聘，不能据此推定已经减员或截至9月仍维持同样招聘速度；没有可比的工程师自愿离职率。[21] |

## 国内大厂对照


| 公司与资料时点 | AI 软件工程实践与效果 | 人员与组织变化 |
| --- | --- | --- |
| **阿里巴巴**；2026年3月至5月事件，产品页9月19日检索 | 官方介绍通义灵码具备工程检索、多文件编辑、终端执行和编程智能体能力；自5月20日起更名 **Qoder CN**。本次未取得可归因的集团内部净提效测量。[22][23] | 媒体转述3月5日 CEO 内部信批准林俊旸辞职；阿里同时表示千问团队稳定、否认“集体离职”。不把报道的动机分析当作已证实的离职原因。[24] |
| **腾讯**；2025研发数据于2026-06-05报道；人数资料8月12日 | 据《腾讯2025研发大数据报告》的媒体转述：超过90%的工程师使用 CodeBuddy，50%新增代码由 AI 辅助生成；微信支付需求交付周期缩短31%。同时涉及 WeDev 流程改造，未分离 AI 的独立因果贡献。[25] | 6月末员工115,927人，上年同期111,221人；上半年总酬金645亿元，同比略降。期末人数与半年酬金的口径不同，不能据此判定个体降薪或裁员。[26] |
| **字节跳动**；产品及招聘页2026-09-19检索 | TRAE 官方已提供 TraeCode、SOLO及 Coding Agent，覆盖规划、编码、调试、测试等；这里只证明产品能力和供给，未取得字节内部采用率或净提效的可靠测量。[31] | 官方招聘页存在 TRAE 核心研发团队的 AI Coding Tools 机器学习工程师职位，涉及训练、量化、部署和 GPU 集群。职位存在是招聘需求信号，不是实际新增人数。[32] |
| **百度**；2026 Q2财报8月18日，产品页9月19日检索 | Comate 官方介绍覆盖规划、编码、单测、调试、审查等开发流程。本次未取得可靠的2026公司级净提效测量；不把历史代码占比或官网功能说明改写成最新生产率。[27] | 公告将研发费用同比下降、环比上升主要归于人员相关费用变化；不能直接换算为 AI 替代人数或关键人才流失。[28] |
| **华为**；产品2026-02-26；研发人数截至2025年末 | 华为云发布“码道”CodeArts代码智能体公测版，覆盖代码生成、研发问答、单测等；官方称 Codebase 索引在同等任务下可省30% token。这是特定产品声明，不是全公司研发 ROI 或内部采用率。[29] | 官方2025年报披露研发人员114,000人，占员工53.7%，统计日为2025年12月31日；不能当成2026人数或直接推断裁员。[30] |

## 论文与咨询研究

| 研究与时间 | 主要发现 |
| --- | --- |
| **[BCG《The Widening AI Value Gap》](https://www.bcg.com/publications/2025/are-you-generating-value-from-ai-the-widening-gap)**，2025-09-30 [1] | 其研究将约5%的企业归为获得显著价值的领先组，35%正在规模化并开始创造价值，60%尚未获得实质性收益。 |
| **[Cui 等：三家企业的软件开发随机试验](https://www.microsoft.com/en-us/research/publication/the-effects-of-generative-ai-on-high-skilled-work-evidence-from-three-field-experiments-with-software-developers/)**，2025-06 [2] | 在 Microsoft、Accenture 和一家匿名 Fortune 100 企业的4,867名开发者中，合并估计显示工具使用者完成任务数增加26.08%，标准误10.3个百分点；经验较少者收益更大。 |
| **METR：[早期随机试验](https://metr.org/blog/2025-07-10-early-2025-ai-experienced-os-dev-study/)与[后续更新](https://metr.org/blog/2026-02-24-uplift-update/)**，2025-07-10、2026-02-24 [3][4] | 早期16名熟悉项目的开发者、246项任务中，使用当时工具耗时增加19%。后续57名开发者、800多项任务的原始估计转向提速，但作者明确认为选择偏差、退出和并行工作计时问题使结果不可靠。 |
| **[METR 技术工作者调查](https://metr.org/blog/2026-05-11-ai-usage-survey/)**，2026-05-11 [5] | 349名受访者自报工作价值提高至原来的1.4至2倍，速度中位数为3倍；便利抽样，邮件响应率约2%。 |
| **DORA：[2025报告](https://dora.dev/research/2025/dora-report/)与[AI开发 ROI 框架](https://cloud.google.com/resources/content/dora-roi-of-ai-assisted-software-development)** [6][7] | AI 往往放大原有组织的优点和缺点；官方 ROI 介绍强调初期学习成本、减少返工、把释放的产能再投入，而非简单减员。 |
| **[BCG《AI Transformation Is a Workforce Transformation》](https://www.bcg.com/publications/2026/ai-transformation-is-a-workforce-transformation)**，2026-02-04 [8] | 基于其2025年高管调查，领先企业计划对超过50%的员工进行 AI 技能培训，落后组约20%；强调岗位、管理和学习机制一起调整。 |
| **[DX 工程预算持有者调查](https://getdx.com/blog/how-are-engineering-leaders-approaching-2026-ai-tooling-budget/)**，2025-10-15，面向2026预算 [38] | 50名预算持有者调查中，接近一半为 AI 工具预留工程预算的1%–3%；文中称1,000美元／开发者／年成为不少公司的目标。另一项275名工程负责人调查中，38.4%自报2025年支出为101–500美元／开发者／年。 |
| **[Accenture《Pulse of Change》](https://www.accenture.com/en/insights/pulse-of-change-september-2025)**，2025-09-05 [39] | 2025年5至6月分别调查3,000名大型企业高管和3,000名员工：85%的高管计划当年增加 AI 投资，63%的组织正在投资 AI Agent，但仅20%从头重构流程；86%的高管称在为 Agentic AI 准备员工，同时75%承认变化速度超过培训能力。 |
| **[Deloitte《The State of AI in the Enterprise 2026》](https://www.deloitte.com/ie/en/issues/generative-ai/state-of-ai-in-enterprise.html)**，2026-01-21 [40] | 2025年8至9月调查24国3,235名直接参与 AI 项目的业务与 IT 高管：获准使用 AI 工具的员工比例由不足40%升至约60%；只有25%的受访者称至少40%的 AI 试点已进入生产，25%称 AI 已带来转型性影响，30%在围绕 AI 重构关键流程。 |
| **[PwC《2026 Global AI Jobs Barometer》](https://www.pwc.com/gx/en/issues/artificial-intelligence/job-barometer/2026/2026-global-ai-jobs-barometer-global-findings.pdf)**，2026-06 [41] | 分析六大洲逾10亿条招聘广告，并用 ORBIS 数据比较企业2018至2024／25年的每员工营业额。AI 暴露度最高组的企业生产率增长比最低组高40%，员工人数增长为52%对36%，工资增长为24%对17%；AI 专业岗位发布量在2024至2025年增长68.9%。 |

## 来源与链接

以下链接对应正文编号，均以2026-09-19为检索截止日。研究结论、企业自报、财报和媒体报道分别标注；报道中的高管发言仍按企业自报对待。付费页仅使用可访问的公开部分，未读到的报告全文不冒充已核验。动态产品和招聘页可能在此后变化。

1. **咨询研究，2025-09-30**：BCG，[The Widening AI Value Gap](https://www.bcg.com/publications/2025/are-you-generating-value-from-ai-the-widening-gap)。用于5%／35%／60%的企业分组；不等同于项目失败率。
2. **实证论文，2025-06，预印本入口**：Cui、Demirer、Jaffe、Musolff、Peng、Salz，[The Effects of Generative AI on High-Skilled Work: Evidence from Three Field Experiments with Software Developers](https://www.microsoft.com/en-us/research/publication/the-effects-of-generative-ai-on-high-skilled-work-evidence-from-three-field-experiments-with-software-developers/)。Microsoft Research 论文页面，列出三家企业、样本量和合并估计。
3. **随机试验，2025-07-10，已被作者标记为历史结果**：METR，[Measuring the Impact of Early-2025 AI on Experienced Open-Source Developer Productivity](https://metr.org/blog/2025-07-10-early-2025-ai-experienced-os-dev-study/)。
4. **后续实证与方法修订，2026-02-24**：METR，[We Are Changing Our Developer Productivity Experiment Design](https://metr.org/blog/2026-02-24-uplift-update/)。用于解释为什么不沿用旧结果，也不把后续点估计当成确定结论。
5. **自报调查，2026-05-11**：METR，[Measuring the Self-Reported Impact of Early-2026 AI on Technical Worker Productivity](https://metr.org/blog/2026-05-11-ai-usage-survey/)。调查实施于2026年2至4月。
6. **行业研究，2025版**：DORA，[State of AI-assisted Software Development 2025](https://dora.dev/research/2025/dora-report/)。官方概要及报告入口。
7. **行业方法框架，介绍页未标明发布日期，2026-09-19检索**：Google Cloud／DORA，[ROI of AI-assisted Software Development](https://cloud.google.com/resources/content/dora-roi-of-ai-assisted-software-development)。本调研引用公开介绍页，不声称已读取需填写表单下载的完整报告。
8. **咨询研究解读，2026-02-04**：BCG，[AI Transformation Is a Workforce Transformation](https://www.bcg.com/publications/2026/ai-transformation-is-a-workforce-transformation)。基于其2025年高管调查及咨询经验。
9. **官方财报，2026-07-29**：Microsoft，[FY2026 Q4 Earnings](https://www.microsoft.com/en-us/Investor/earnings/FY-2026-Q4/press-release-webcast)。仅用于自愿退休、遣散及相关人事说明，不用集团财务金额替代员工 AI 工具费用。
10. **高管自报，2026-04-22**：Google，[Sundar Pichai at Google Cloud Next 2026](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/cloud-next-2026-sundar-pichai/)。用于75%新增代码、人工批准及特定迁移案例，不能代表全公司的净提效。
11. **官方组织公告，2026-08**：Google，[The Next Chapter of Our AI Momentum](https://blog.google/company-news/inside-google/message-ceo/next-chapter-ai-momentum/)。用于 DeepMind 领导安排及 Jeff Dean、Sanjay Ghemawat 的独立公司计划与合作关系。
12. **股东信、管理层自报，2026-04-09**：Amazon，[CEO Andy Jassy's 2025 Letter to Shareholders](https://www.aboutamazon.com/news/company-news/amazon-ceo-andy-jassy-2025-letter-to-shareholders)。信名为2025年度，发布日期为2026年；Mantle团队与时间为自报，传统人力需求是估计。
13. **官方人事公告，2026-01-28**：Amazon／Beth Galetti，[Update on Our Organization](https://www.aboutamazon.com/news/company-news/amazon-layoffs-corporate-jan-2026)。约16,000个岗位、内部求职安排；不是 AI 替代人数统计。
14. **媒体报道与公司回应，2026-02-20**：Reuters，[Amazon's Cloud Unit Hit by Outage Involving AI Tools in December](https://www.reuters.com/business/retail-consumer/amazons-cloud-unit-hit-by-least-two-outages-involving-ai-tools-ft-says-2026-02-20/)；本次读取的[路透稿公开转载](https://www.marketscreener.com/news/amazon-s-cloud-unit-hit-by-outage-involving-ai-tools-in-december-ce7e5dddd981f225)。FT指称的13小时、Kiro决策，与 Amazon 对范围及用户错误的回应分别归属，不混为共同确认事实。
15. **企业技术自报，2026-03-17**：Meta Engineering，[Ranking Engineer Agent: An Autonomous AI System Accelerating Meta's Ads Ranking Innovation](https://engineering.fb.com/2026/03/17/developer-tools/ranking-engineer-agent-rea-autonomous-ai-system-accelerating-meta-ads-ranking-innovation/)。仅适用于文中排序模型研发部署案例。
16. **官方财报，2026-07-29**：Meta，[Second Quarter 2026 Results](https://investor.atmeta.com/investor-news/press-release-details/2026/Meta-Reports-Second-Quarter-2026-Results/default.aspx)。用于员工总数和约8,000名受裁员影响员工仍计入人数的注释，不用资本开支代替员工工具费用。
17. **媒体报道、高管提议，2026-03-20**：CNBC，[Nvidia's Huang Pitches AI Tokens on Top of Salary as Agents Reshape How Humans Work](https://www.cnbc.com/2026/03/20/nvidia-ai-agents-tokens-human-workers-engineer-jobs-unemployment-jensen-huang.html)。预算设想不是已经发生的年度支出，也不是用 token 替代薪资。
18. **媒体依据年报，2026-06-22／23，时区不同**：Reuters，[Oracle Workforce Shrinks by About 21,000 Employees Amid AI Adoption](https://www.reuters.com/business/world-at-work/oracle-workforce-shrinks-by-about-13-2026-06-22/)。用于5月末人数和披露的多种重组原因；不是21,000人全部被 AI 替代的证据。
19. **官方招聘承诺，2026-05-06**：Salesforce，[Salesforce Commits to Hiring 1,000 AI-Native Grads](https://www.salesforce.com/news/stories/hiring-ai-native-graduates/)。包含毕业生与实习生，覆盖多种岗位，不能当成已录用1,000名软件工程师。
20. **企业工程自述，2026-09-16**：Salesforce／Jayesh Govindarajan、Silvio Savarese，[Why We Post-Trained Our Own Reasoning Model](https://www.salesforce.com/news/stories/why-we-post-trained-our-own-reasoning-model/)。用于 Koa 的训练、内部运行和客户试点阶段，不采用未披露方法的性能宣传作独立实证。
21. **二手媒体报道，2026-04-23**：AI Magazine，[Why Uber Has Already Burned Through Its AI Budget](https://aimagazine.com/news/why-uber-has-already-burned-through-its-ai-budget)。转述 CTO 和 The Information；本次未读取后者原始付费报道。报道给出预算用尽及500–2,000美元／人／月的开销区间，但未提供代表性样本、公司预算绝对额或实际结算总额。
22. **官方产品介绍，2026-09-19检索**：阿里云，[通义灵码产品页](https://lingma.aliyun.com/)。页面已提示升级至 Qoder CN；功能介绍不等于内部实测收益。
23. **官方产品变更公告，2026-05-08**：阿里云，[智能编码助手通义灵码更名为 Qoder CN](https://www.aliyun.com/notice/detail?notice-id=118234)。生效日为2026年5月20日。
24. **媒体转述与评论，2026-03-10**：21财经转载连线Insight，[从林俊旸离职，看阿里内部技术理想与商业现实的深层博弈](https://www.21jingji.com/article/20260310/herald/7a41cd87204ce89c0d722ddd7f11d65b.html)。仅引用批准辞职及公司否认集体离职的报道，不把作者对动机的分析当成事实。
25. **媒体转述企业研发报告，2026-06-05**：证券时报网转载界面新闻，[腾讯 AI 编程及研发效能报道](https://www.stcn.com/article/detail/3945575.html)。引用的是《腾讯2025研发大数据报告》相关采用率、代码占比和微信支付交付周期，不是2026全员新测量。
26. **财报转述，2026-08-12**：华尔街见闻／张雅琦，[腾讯Q2业绩与员工数据](https://wallstreetcn.com/articles/3779275)。仅引用员工总数及酬金；[腾讯官方业绩入口](https://www.tencent.com/zh-cn/investors/results/)可回查，本次未成功抽取官方PDF全文。
27. **官方产品介绍，2026-09-19检索**：百度，[Comate 文心快码](https://comate.baidu.com/zh)。仅引用开发流程和功能，未以百科、转载的旧采用数据充当最新效率。
28. **官方财报，2026-08-18**：百度，[Second Quarter 2026 Results](https://ir.baidu.com/news-releases/news-release-details/baidu-announces-second-quarter-2026-results/)。仅用于人员相关费用的变化说明，不把总研发费用当员工 AI 费用。
29. **官方产品公告，2026-02-26**：华为云，[华为云码道（CodeArts）代码智能体公测版正式发布](https://www.huaweicloud.com/news/2026/20260226150052593.html)。30% token节约为产品方特定任务声明，不是全公司人力成本下降。
30. **官方年报，2025年度、2026年发布**：华为，[2025 Annual Report](https://www.huawei.com/en/annual-report/2025)。用于截至2025年12月31日研发人数及其占比，不当成2026实时人数。
31. **官方产品介绍，2026-09-19检索**：TRAE，[TraeCode 与 SOLO](https://www.trae.cn/)。产品功能不代表字节内部采用率；公司归属与团队信息同时参照[32]。
32. **官方招聘页，2026-09-19检索，未标发布日期**：ByteDance Careers，[Machine Learning Engineer, AI Coding Tools](https://joinbytedance.com/search/7571650125270370613)。TRAE核心研发、模型训练及部署职责；只作为招聘需求信号，不按搜索职位数推算招聘人数。
33. **记者采访，2026-06-13**：经济观察报／刘思璇，[大厂Token不再“管够”：腾讯开始限额，字节可部分报销](https://www.eeo.com.cn/2026/0613/913745.shtml)。用于腾讯、阿里部门额度，字节部分部门报销规则，腾讯个人额度耗尽及米哈游负责人分享的单次消耗。报道没有公司级采购账单或实付总额，不将受访部门推广为全员政策。
34. **记者采访，2026-06-10**：第一财经／刘佳，证券时报网转载，[全面拥抱AI后，大厂终于给Token“算账”了](https://www.stcn.com/article/detail/3953332.html)。用于匿名部门一个月5万元的消耗案例，以及腾讯按需求动态分配、预算增长但未给绝对额的报道。匿名公司不作身份推断。
35. **媒体转述，2026-08-04**：The Next Web，[Microsoft Tells Employees to Stop Tokenmaxxing, Sets Division-Level AI Budgets](https://thenextweb.com/news/microsoft-tokenmaxxing-ai-spending-limits)。转述内部邮件、7月部门级上限及工程师月开销数量级；原始报道为404 Media，[Microsoft Tells Engineers 'Tokenmaxxing Is Not What We Are Optimizing For'](https://www.404media.co/microsoft-tells-engineers-tokenmaxxing-is-not-what-we-are-optimizing-for/)，本次只读取其公开开头，细节依据 TNW，未读取付费全文。
36. **媒体转述，2026-06-17**：The Next Web，[The Tokenmaxxing Era Is Over. Now Companies Are 'Tokenminimizing'](https://thenextweb.com/news/tokenminimizing-companies-cap-employee-ai-spending)。仅用于 Uber每人每工具每月1,500美元上限、Meta收紧使用及Amazon用量管理。所引 The Information原文需订阅，本次未读取全文；不采用该文中归属有疑点的微软500–2,000美元数字，也不把预测视为已付。
37. **高管访谈报道，2026-03-20，页面为GMT+8**：Business Insider／Lee Chong Ming，[Jensen Huang on Engineers' Annual Token Spending](https://www.businessinsider.com/jensen-huang-500k-engineers-250k-ai-tokens-nvidia-compute-2026-3)。用于年薪50万美元工程师的25万美元年度用量期望及面对团队20亿美元问题的回应；没有预算审批或实际付款证明。
38. **供应商预算调查，2025-10-15**：DX，[How Are Engineering Leaders Approaching 2026 AI Tooling Budgets?](https://getdx.com/blog/how-are-engineering-leaders-approaching-2026-ai-tooling-budget/)。分别包含50名工程预算持有者的2026规划调查及另一项275名负责人对2025支出的自报，不是13家大厂的具名采购数据。
39. **咨询调查，2025-09-05**：Accenture，[Pulse of Change](https://www.accenture.com/en/insights/pulse-of-change-september-2025)。两项全球调查各有3,000名受访者，覆盖18个国家和22个行业；高管来自年收入超过5亿美元的组织。用于投资意愿、Agent投入、流程重构及培训差距，不作为实际支出或独立效果评估。
40. **咨询调查，2026-01-21**：Deloitte，[From Ambition to Activation: Organizations Stand at the Untapped Edge of AI's Potential](https://www.deloitte.com/us/en/about/press-room/state-of-ai-report-2026.html)；[报告与方法页](https://www.deloitte.com/ie/en/issues/generative-ai/state-of-ai-in-enterprise.html)。调查于2025年8至9月进行，覆盖24国3,235名直接参与企业 AI 项目的董事至高管级受访者；用于工具覆盖、试点生产化和流程重构的自报结果。
41. **咨询研究，2026-06**：PwC，[2026 Global AI Jobs Barometer: Global Findings](https://www.pwc.com/gx/en/issues/artificial-intelligence/job-barometer/2026/2026-global-ai-jobs-barometer-global-findings.pdf)。招聘数据覆盖2012至2025年；企业生产率采用 ORBIS 数据，以2018至2024／25年的每员工营业额衡量。用于生产率、员工人数、工资和 AI 岗位变化的关联性结果，不作为因果估计。
