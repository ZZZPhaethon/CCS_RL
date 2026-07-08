# Northern Lights 项目数据汇总

口径日期：2026-06-10。只记录能从给定 PDF 或官方/权威网页追溯到出处的数据；未找到可靠出处的字段不填数值。单位按来源保留，必要时只做明确的四则运算并标注为“计算”。

## 1. 项目范围与能力

| 项目 | 已确认数据 | 来源 |
|---|---:|---|
| 业务范围 | 工业源 CO2 捕集后液化，由船运至挪威西部 Øygarden 接收终端，中间储存后经管道送至北海海底储层永久封存。 | Northern Lights 官网 What we do；Northern Lights 官网 About Longship |
| Phase 1 年处理能力 | 1.5 Mt CO2/年。 | Northern Lights 官网 What we do；`Northern-Lights-4061-SF8-Arsrapport-2023.pdf`, p.10, p.17 |
| Phase 2 目标能力 | 至少 5 Mt CO2/年；2025 年 FID，预计 2028 年下半年投运。 | Northern Lights 2025-03-27 Phase 2 新闻；Northern Lights 2026-03-17 Phase 2 新闻 |
| Longship 保留给两个挪威捕集项目的能力 | Heidelberg Materials Brevik + Hafslund Celsio 合计 0.8 Mt CO2/年。 | Northern Lights 官网 How to store CO2 |
| Phase 1 25 年注入量 | 37.5 Mt CO2。 | `Northern-Lights-4061-SF8-Arsrapport-2023.pdf`, p.17 |
| 价值链生命周期净减排 | 按 Northern Lights 2023 LCA：127.8 Mt CO2 存储，3.3 Mt CO2e 生命周期排放，净减排 124.5 Mt CO2，净减排率 97.4%。 | `Northern-Lights-4061-SF8-Arsrapport-2023.pdf`, p.18 |

## 2. 排放源、阶段、坐标与捕集信息

口径说明：Northern Lights 官网在 2026 年 Phase 2 更新中称已有 5 个 industrial emitters secured long-term agreements。这里的“5 个”按客户/工业主体口径统计；若按物理排放点统计，Ørsted Kalundborg Hub 包含 Asnæsværket 和 Avedøreværket 两个电厂，因此为 6 个排放点。坐标为 OpenStreetMap/Nominatim 按厂区名称匹配得到的 WGS84 经纬度，用于 GIS/建模前建议再核对边界点或装船点。

| 口径 | Emitter / 物理点 | 阶段归类 | 坐标 WGS84, lat, lon | 排放量或合同量 | 捕集技术 | 捕集比例 | 来源与说明 |
|---|---|---|---:|---:|---|---:|---|
| 客户 1 / 物理点 1 | Heidelberg Materials / Norcem Brevik cement factory | Phase 1 / Longship 保留容量 | 59.064393, 9.695636 | 水泥厂烟气 CO2 约 0.8 Mt/年；捕集 0.4 Mt/年。 | 后燃烧液体胺法；Brevik CCS 官网列为 amine, post-combustion capture；战略伙伴 SLB Capturi；Norcem FEED 还写明 Aker Solutions ACC amine-based 技术。 | 约 50%（0.4/0.8，计算）。 | Gassnova 2025 cost reduction report, p.7；Brevik CCS Facts and FAQ；Norcem FEED, p.8, p.13, p.151；坐标匹配 OSM: Norcem, Brevik。 |
| 客户 2 / 物理点 2 | Hafslund Celsio Klemetsrud waste-to-energy plant, Oslo | Phase 1 / Longship 保留容量 | 59.840556, 10.836344 | WtE 工厂约 0.4 Mt CO2/年；计划捕集 350,000 t CO2/年；其中约 60% 为生物源碳，约 200,000 t/年生物源 CO2 可移出碳循环。 | 公开网页未给最终供应商；Fortum Oslo Varme 试验报告显示 Klemetsrud 试验装置为胺法，CO2 removal efficiency 目标 90%+，典型 80-100%。 | 最高约 87.5%（350,000/400,000，计算；因来源写“up to”，按上限处理）。 | CCS Norway Hafslund Celsio 页面；Fortum Oslo Varme pilot test report, p.6；坐标匹配 OSM: Klemetsrud forbrenningsanlegg。 |
| 客户 3 / 物理点 3 | Yara Sluiskil ammonia production | 商业跨境客户；官网未把该条单独标为 Phase 2 | 51.276836, 3.852485 | Northern Lights 合同目标：从 2026 年起捕集并封存 800,000 t CO2/年。 | 未在已查可靠来源中找到具体捕集技术。 | 未找到可靠公开比例。 | Northern Lights 2023-11-20 Yara 新闻；`Northern-Lights-4061-SF8-Arsrapport-2023.pdf`, p.30；坐标匹配 OSM: Yara Sluiskil B.V.。 |
| 客户 4 / 物理点 4 | Ørsted Asnæsværket biomass power station | 商业跨境客户；Ørsted capture hub 的一个物理点 | 55.662051, 11.079183 | Ørsted Kalundborg Hub 合同总量的一部分：Asnæs + Avedøre 合计 430,000 t 生物源 CO2/年；2026-01-01 起，10 年。 | Ørsted 将建设 CO2 capture hub；已查来源未给具体技术。 | 未找到可靠公开比例。 | Northern Lights 2023-05-15 Ørsted 新闻；`Northern-Lights-4061-SF8-Arsrapport-2023.pdf`, p.32；坐标匹配 OSM: Asnæsværket。 |
| 客户 4 / 物理点 5 | Ørsted Avedøreværket power station | 商业跨境客户；Ørsted capture hub 的一个物理点 | 55.602575, 12.487710 | 同属 Ørsted Kalundborg Hub 合同；430,000 t/年为两座电厂合计，不应与上一行重复相加。 | Ørsted 将建设 CO2 capture hub；已查来源未给具体技术。 | 未找到可靠公开比例。 | Northern Lights 2023-05-15 Ørsted 新闻；`Northern-Lights-4061-SF8-Arsrapport-2023.pdf`, p.32；坐标匹配 OSM: Avedøreværket。 |
| 客户 5 / 物理点 6 | Stockholm Exergi BECCS / existing heat and power biomass plant, Stockholm | Phase 2 触发性商业客户；2028 年起 | 59.352320, 18.103059 | 最多 900,000 t 生物源 CO2/年，15 年，从 2028 年开始。 | BECCS；已查 Northern Lights 来源未给捕集工艺细节。 | 未找到可靠公开比例。 | Northern Lights 2025-03-27 Phase 2 新闻；2026 Phase 2 更新称其为 launching Phase 2 的 key catalyst；坐标匹配 OSM: Värtaverket。 |

Northern Lights 接收与中间储存 hub 不属于 emitter，位置为 Naturgassparken / Energiparken, Øygarden。OpenStreetMap/Nominatim 匹配坐标为 60.553892, 4.882342。该 hub 接收船运液态 CO2，在陆上储罐中间储存后经管道送至海底储层。

## 3. 中间储存、码头与装卸（不含管道）

### 3.1 按 emitter 的 buffer / intermediate storage

| Emitter / 客户 | 是否有本地或发运端 buffer/storage 的公开证据 | 已公开数据 | 未公开/不确定项 | 来源 |
|---|---|---|---|---|
| Heidelberg Materials / Norcem Brevik | 是。Brevik CCS 包含 CO2 conditioning、intermediate storage 和 loading；CO2 tank farm 与 ship loading 是公开 FEED 的一部分。 | 捕集厂本地提供 local intermediate storage；CO2 tank farm 部分位于 Tangen Eiendom 区域；液态 CO2 装船流量 800 t/h（间歇，仅装船时）；储罐/管线/船的热输入估计产生 1 t/h 蒸发气，船舱置换气返回中间储罐。 | 未在公开 redacted FEED 中找到储罐数量、单罐容量或总容量。 | Norcem FEED, p.42-p.43, p.60, p.78 |
| Hafslund Celsio / Klemetsrud | 是，但公开资料显示 buffer 在 Oslo CCS Terminal，而不是直接等同于 Klemetsrud 焚烧厂厂内储罐。 | Oslo CCS Terminal 有 truck unloading、temporary storage of liquid CO2、ship loading；completion strategy 明确写到“三个 intermediate storage tanks”；CO2 tanks 当前设计含 110 mm insulation；truck unloading station 有两个卸车站。 | 未在公开 FEED 主报告中找到单罐容量或总容量；容量可能在单独的 process datasheet / tank size study 中，但主报告未披露数值。 | Hafslund Celsio 2025 FEED, p.3, p.23, p.26, p.87 |
| Yara Sluiskil | 有液化并从荷兰发运的公开描述，但未找到储罐/buffer 细节。 | Northern Lights-Yara 协议写 CO2 将液化并从荷兰船运至挪威大陆架永久封存。 | 未找到公开储罐数量、容量、是否专设 buffer tank farm。 | Northern Lights 2023-11-20 Yara 新闻；`Northern-Lights-4061-SF8-Arsrapport-2023.pdf`, p.30 |
| Ørsted Asnæs + Avedøre / Kalundborg Hub | 有 hub 发运概念的公开描述，但未找到储罐/buffer 细节。 | Ørsted 将建设 CO2 capture hub；两座电厂合计 430,000 t/年，CO2 将由 Northern Lights 运输和封存。 | 未找到公开资料确认每座电厂分别有储罐；也未找到 Kalundborg Hub 的储罐数量/容量。 | Northern Lights 2023-05-15 Ørsted 新闻；`Northern-Lights-4061-SF8-Arsrapport-2023.pdf`, p.32 |
| Stockholm Exergi BECCS | 有 BECCS 客户和起始年份公开信息；未找到与 Northern Lights 接口相关的储罐/buffer 细节。 | 2028 年起最多 900,000 t/年，15 年。 | 未找到公开储罐数量、容量或装船 buffer 信息。 | Northern Lights 2025-03-27 Phase 2 新闻 |

结论：不能说“每个 emitter 都有公开披露的本地储罐容量”。可以确定的是，船运链条需要捕集/发运端有缓冲能力；Brevik 和 Celsio/Oslo CCS Terminal 有明确公开证据。Yara、Ørsted 和 Stockholm Exergi 只有液化/发运或合同量信息，储罐容量细节未找到可靠公开出处。

| 位置 | 已确认数据 | 来源 |
|---|---|---|
| Øygarden 接收终端 | 位于 Energiparken / Naturgassparken, Øygarden；包括进口码头、CO2 储罐、液态 CO2 装卸臂/设备、工艺系统、质量控制、变电站、行政综合体。 | Northern Lights 官网 What we do；CCS Norway Northern Lights 页面；`1-Northern-Lights-Summit-Cristel-Lambton-Facilities.pdf`, p.8-p.9 |
| 接收终端储存条件 | 13-18 barg，-30 至 -20 deg C；常规操作远程控制；Phase 1 设计基准 1.5 Mt/年液态 CO2。 | `1-Northern-Lights-Summit-Cristel-Lambton-Facilities.pdf`, p.8 |
| 接收终端储罐 | 2023 年已完成全部 12 个陆上储罐安装；每个储罐高 32.5 m，容量接近 700 t CO2；12 个罐合计接近 8,400 t CO2（由 12 x 700 计算，原文为 nearly 700 tonnes per tank）。 | `Northern-Lights-4061-SF8-Arsrapport-2023.pdf`, p.22 |
| 捕集端本地储存 | 2020 设施页说明：Norcem Brevik 和 Fortum/Celsio Klemetsrud 捕集的 CO2 在各自码头本地储存；储存量需覆盖船每 4 天到港一次并带全链条不确定性缓冲。 | `1-Northern-Lights-Summit-Cristel-Lambton-Facilities.pdf`, p.5 |
| Brevik 装船流量 | Norcem FEED 简化流程图列出“Liquid, intermittent - during ship loading only”为 800 t/h；未找到由此确认的总装船时长。 | Norcem FEED, p.42 |
| Oslo CCS Terminal | 2025 FEED 为 Celsio 的 Oslo CCS Terminal；包括 CO2 truck unloading station、CO2 storage、export facility to ship；truck unloading station 有 2 个卸车站，分别配 2 个和 1 个卸车臂。 | Hafslund Celsio 2025 FEED, p.8, p.25 |

## 4. 船舶

| 项目 | 已确认数据 | 来源 |
|---|---|---|
| Phase 1 船舶数量与容量 | 到 2026 年初，Northern Pioneer、Northern Pathfinder、Northern Phoenix 三艘 7,500 m3 姊妹船已交付；第四艘相同船 2026 年交付/完工中。 | Northern Lights 2026-01-29 船队扩张新闻 |
| Phase 1 单船容量 | 7,500 m3 液态 CO2。 | Northern Lights 2026-01-29 船队扩张新闻；`Northern-Lights-4061-SF8-Arsrapport-2023.pdf`, p.21, p.53；`1-Northern-Lights-Summit-Cristel-Lambton-Facilities.pdf`, p.5-p.6 |
| Phase 1 早期设计参数 | 一座捕集厂一艘船；装船压力 13-15 barg，卸船压力 13-18 barg；平衡温度约 -30 deg C；船长约 130 m，型宽约 20 m，吃水约 8.5 m。 | `1-Northern-Lights-Summit-Cristel-Lambton-Facilities.pdf`, p.5-p.7 |
| Phase 2 新船 | 船队从 4 艘扩至 8 艘；新船计划 2028-2029 年交付。2026 年 1 月新闻称前三艘新船各 12,000 m3。 | Northern Lights 2026-03-17 Phase 2 新闻；Northern Lights 2026-01-29 船队扩张新闻 |
| 船速、实际装船/卸船时长 | 未找到 Northern Lights 官方项目值。Fraga et al. 2024 附录 A1 的 27.8 km/h 船速和 26/50/80 h 装卸时间是优化模型参数，不是 Northern Lights 船舶实测或官方设计值，不能作为项目事实使用。 | `Fraga_2024_multi_period_multi_objective_optimisation.pdf`, p.22-p.23 |

## 5. 管道与海底输送设施

| 项目 | 已确认数据 | 来源 |
|---|---|---|
| 管道功能 | 从 Øygarden 接收终端/中间储罐向 Aurora 海底储层输送 CO2。 | Northern Lights 官网 What we do；`1-Northern-Lights-Summit-Cristel-Lambton-Facilities.pdf`, p.5 |
| 管道输送相态 | 设施页明确写为“single phase (liquid) CO2”；即管道设计目标是单相液态 CO2 输送，不是气态 CO2 输送。 | `1-Northern-Lights-Summit-Cristel-Lambton-Facilities.pdf`, p.5 |
| 管道长度 | 公开来源存在差异：2023 年报和 2020 概览页写 110 km；2020 设施页详细管道页写 100.4 km。 | `Northern-Lights-4061-SF8-Arsrapport-2023.pdf`, p.17, p.19；`1-Northern-Lights-Summit-Cristel-Lambton-Facilities.pdf`, p.5, p.10 |
| 管径/材料 | 2020 概览页写 12 1/4 inch、未保温；详细管道页写 12 inch OD、100.4 km C-Mn pipeline。 | `1-Northern-Lights-Summit-Cristel-Lambton-Facilities.pdf`, p.5, p.10 |
| 管道设计能力 | Øygarden 到永久储层的 CO2 管道按 5 Mt/年设计。 | Gassnova 2025 cost reduction report, p.8 |
| 管道施工/路由细节 | HDD landfall；图中标注 HDD length 539 m；1 PLEM、2 个 in-line tee、1 个 spool；27 处与其他管道/电缆交叉。 | `1-Northern-Lights-Summit-Cristel-Lambton-Facilities.pdf`, p.10-p.11 |
| 海底设施 | 单井卫星结构；1 条 36 km fluid umbilical（MEG + hydraulic），1 条 36 km DC/FO（power + signal），接入 Oseberg A；卫星结构安装在 Eos，水深 300 m。 | `1-Northern-Lights-Summit-Cristel-Lambton-Facilities.pdf`, p.12-p.13 |

说明：CO2 可以用单相液态在管道中输送，前提是沿线压力和温度始终保持在液相区并留有裕度，避免出现气液两相、干冰或跨越饱和线。Northern Lights 公开设施页的设计表述就是“single phase (liquid) CO2”。到储层条件后，CO2 会处于高压高温的致密/超临界状态；这不改变管道段公开描述为单相液态输送。

## 6. 地质储层与永久封存

| 项目 | 已确认数据 | 来源 |
|---|---|---|
| 储层位置 | Aurora 储层，EL001，Troll 以南；Eos 确认井 2020 年 3 月成功钻完；储层位于海床以下约 2,600 m。 | Northern Lights 官网 What we do；Northern Lights 官网 About Longship；CCS Norway Northern Lights 页面 |
| 注入井/储层状态 | 2022 年 Phase 1 注入井 A-7 AH 和备用井 C-1 H 的钻完井计划完成；CO2 注入深度约 2-3 km，储层压力约 200-300 bar，温度约 100 deg C。 | Northern Lights 官网 What we do；`1-Northern-Lights-Summit-Cristel-Lambton-Facilities.pdf`, p.5 |
| Aurora 地质 | Johansen 和 Cook 砂岩为主储层，Drake Formation 为主要封盖层；Marashi 模型表给 Johansen/Cook 厚度 173 m、孔隙度 7.3-31.4%、水平渗透率 0.1-500 mD、Eos 井深 2700 m、温度 98 deg C、盐度 15%。 | `Marashi - Northern Lights Project Aurora Model Investigation with Sensitivity Studies and Using Different Sim.pdf`, p.20-p.22, p.70-p.71 |

## 7. 成本信息

| 成本项 | 已确认数据 | 来源 |
|---|---:|---|
| Longship 总成本 | NOK 25.1 billion，含投资和 10 年运营。 | CCS Norway Costs 页面；Gassnova 2025 cost reduction report, p.4 |
| 挪威国家承担部分 | NOK 16.8 billion，约占总成本 2/3。 | CCS Norway Costs 页面；Gassnova 2025 cost reduction report, p.4 |
| 2020 投资预算 | Brevik CCS NOK 3.2 billion；Northern Lights NOK 9.1 billion。 | Gassnova 2025 cost reduction report, p.5 |
| 2020 年运营成本估计 | Brevik CCS NOK 119 million/年；Northern Lights NOK 477 million/年。 | Gassnova 2025 cost reduction report, p.5 |
| 2024 秋季投资成本拆分 | Brevik integration NOK 2.2 bn；capture/compression/intermediate storage NOK 2.7 bn；Northern Lights ships NOK 1.3 bn；Øygarden receiving terminal NOK 2.8 bn；permanent storage NOK 4.1 bn；合计 NOK 13.1 bn。 | Gassnova 2025 cost reduction report, p.6-p.7 |
| 国家支持条款 | Heidelberg Materials：约 NOK 1.2 bn 以内投资成本 100% 覆盖，超过部分 75% 覆盖至上限；Northern Lights JV：base investments 80% 覆盖，additional investments 50% 覆盖；符合条件的 10 年运营成本 75-100% 覆盖至上限。 | Gassnova 2025 cost reduction report, p.4 |
| Northern Lights Phase 2 投资 | JV 业主投资 NOK 7.5 billion；包括欧盟 CEF Energy 批准的 EUR 131 million grant。 | Northern Lights 2025-03-27 Phase 2 新闻 |

## 8. 未采用或需谨慎的数据

- `Environmental impact assessment.pdf` 实际是 Project Greensand Future EIA，不是 Northern Lights；未用于 Northern Lights 数据表。
- `Northern-Lights-FEED-report-public-version.pdf` 本地文件为图片型 PDF，文本层为空；本次只用其目录确认章节结构，没有从不可 OCR 的正文中提取数值。相同/相关公开 FEED 和报告优先用 Gassnova/CCS Norway 可检索在线版本或设施 PDF。
- Fraga et al. 2024 的船速、装卸时间、船型规模等用于多周期优化模型，不等于 Northern Lights 官方船舶运行参数；已单独标注。
- Fraga et al. 2024 的 intermediate storage tank 容积、单罐储量、罐数和 LCI 排放因子是作者基于文献/工程假设建立的库存模型结果，不能替代 Northern Lights 年报或 FEED 中披露的实际工程数据。
- Fraga et al. 2021 的 Grenland multi-user intermediate storage facility 是 design / pre-feasibility study；其 2 Mtpa 聚合流量、7/15 bar 情景、船型、储罐容量、成本结果等是场景设计与 MILP 成本优化输入/输出，不应作为 Northern Lights 已建项目事实数据。
- 对 Yara、Ørsted、Stockholm Exergi，本次只确认了合同捕集/封存量与起始年份。未找到可靠来源给出各源总排放量、捕集百分比和具体捕集技术，因此不填。

## 主要网页来源

- Northern Lights What we do: https://norlights.com/what-we-do/
- Northern Lights How to store CO2: https://norlights.com/how-to-store-co2-with-northern-lights/
- Northern Lights About Longship: https://norlights.com/about-the-longship-project/
- Northern Lights Yara agreement: https://norlights.com/news/northern-lights-and-yara-signs-binding-agreement-on-co2-transport-and-storage/
- Northern Lights Ørsted agreement: https://norlights.com/news/northern-lights-enters-into-cross-border-transport-and-storage-agreement-with-orsted/
- Northern Lights Phase 2 / Stockholm Exergi: https://norlights.com/news/northern-lights-is-expanding-capacity-through-commercial-agreement/
- Northern Lights fleet expansion: https://norlights.com/news/northern-lights-expands-the-fleet-with-four-more-co%E2%82%82-ships/
- Northern Lights Phase 2 update 2026: https://norlights.com/news/northern-lights-phase-2-expanding-europes-co%E2%82%82-storage-capacity/
- CCS Norway Longship overview: https://ccsnorway.com/the-project/
- CCS Norway Heidelberg Materials: https://ccsnorway.com/capture-heidelberg-materials/
- Brevik CCS Facts and FAQ: https://www.brevikccs.com/en/facts-and-faq
- CCS Norway Hafslund Celsio: https://ccsnorway.com/capture-hafslund-celsio/
- CCS Norway Northern Lights: https://ccsnorway.com/transport-storage-northern-lights/
- CCS Norway Costs: https://ccsnorway.com/costs/
- Gassnova Norcem FEED: https://gassnova.no/app/uploads/sites/6/2020/07/NC03-NOCE-A-RA-0009-Redacted-FEED-Study-DG3-Report-Rev01-1.pdf
- Gassnova cost reduction report 2025: https://gassnova.no/app/uploads/sites/6/2025/09/Ny080925_ENG_Endelig_Ekstern-rapport-Kostnader.pdf
- Fortum Oslo Varme pilot test report: https://gassnova.no/app/uploads/sites/6/2020/12/Pilot-Plant-Test-Report-Extended-Phase.pdf
