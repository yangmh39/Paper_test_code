# -*- coding: utf-8 -*-
"""
两阶段蒙特卡洛模拟的产线布局鲁棒性后评估 —— 实验脚本（对齐 方法.pdf）
Stage 1: 确定性优化（贪心列表调度作为 MILP 代理，接口处可替换 PuLP/Gurobi）
Stage 2: 蒙特卡洛后评估（Normal 工时 / Normal 疲劳 / 伯努利工位故障 + 收敛判据）

用法:
    python -I experiment.py [--products 60] [--sims 4000] [--seed 0] [--out results.csv]
"""
import argparse
import sys
import numpy as np
import pandas as pd
from scipy import stats

# Windows 控制台中文输出
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ----------------------------------------------------------------------------
# 0. 常量与参数
# ----------------------------------------------------------------------------
COMPONENTS = [
    # name      value_share  base_time  difficulty  posture(0坐1站)  depth
    ("battery",  0.03, 4,   0.8, 0, 0),
    ("screen",   0.12, 10,  1.4, 1, 1),
    ("rear_cam", 0.08, 6,   0.9, 0, 2),
    ("front_cam",0.02, 4,   0.7, 0, 3),
    ("board",    0.20, 18,  2.0, 1, 4),
    ("storage",  0.05, 3,   0.5, 0, 5),
]

# 布局特征参数（需在论文中给出依据/做敏感性校准）
LAYOUTS = {
    "U":  dict(n_ws=6, fatigue_mult=1.10, posture_flex=True),
    "L":  dict(n_ws=6, fatigue_mult=1.00, posture_flex=False),
    "C":  dict(n_ws=6, fatigue_mult=1.05, posture_flex=True),
    "P":  dict(n_ws=8, fatigue_mult=0.90, posture_flex=True),
}
LAYOUT_NAMES = {"U": "U型", "L": "直线", "C": "环型", "P": "并行"}

CAP_TIME = 60.0      # 每工位节拍容量 τ_f（min），工时与疲劳共享同一节拍（与模型一致）
CAP_FATIGUE = 60.0
SLACK = 0.85         # Stage1 标称方案目标利用率，预留扰动缓冲
UNIT_COST = 0.4      # 单位工时处理成本
COARSE_VALUE = 0.6   # 粗拆价值回收率
COARSE_TIME = 0.6    # 粗拆工时比例
SIGMA_E = 0.2        # 疲劳变异系数默认值

METHODS = ["post", "ro", "sp", "mc"]


# ----------------------------------------------------------------------------
# 1. 数据加载：手机规格 -> 产品拆解图
# ----------------------------------------------------------------------------
def load_products(path, n=60, seed=0):
    df = pd.read_csv(path)
    df = df[df["price"].notna() & (df["price"] > 0)].copy()
    # 按价格分三档分层抽样
    df["tier"] = pd.qcut(df["price"], 3, labels=[0, 1, 2])
    rng = np.random.default_rng(seed)
    idx = []
    for t in [0, 1, 2]:
        sub = df[df["tier"] == t]
        k = min(len(sub), int(np.ceil(n / 3)))
        idx += list(rng.choice(sub.index, size=k, replace=False))
    return df.loc[idx].reset_index(drop=True)


def build_candidates(products):
    """所有产品（已按价格降序）的候选任务扁平列表。

    每个零部件有 精细/粗拆 两种互斥模式；pid 越小价格越高、优先级越高。
    """
    cands = []
    for pid, (_, row) in enumerate(products.iterrows()):
        price = float(row["price"])
        for name, share, t0, diff, posture, depth in COMPONENTS:
            value = price * share
            cands.append(dict(pid=pid, name=name, mode="fine", depth=depth,
                              posture=posture, value=value, time=t0,
                              fatigue=t0 * diff))
            cands.append(dict(pid=pid, name=name, mode="coarse", depth=depth,
                              posture=posture, value=value * COARSE_VALUE,
                              time=t0 * COARSE_TIME,
                              fatigue=t0 * diff * COARSE_TIME))
    return cands


# ----------------------------------------------------------------------------
# 2. Stage 1：确定性优化（贪心列表调度，MILP 代理，共享工位容量）
# ----------------------------------------------------------------------------
def solve_stage1(cands, layout, method="post"):
    """共享 W 个工位的容量，从候选任务中贪心选出最优拆解方案。

    method 决定 Stage 1 使用的工时:
      post: 标称工时
      ro  : 盒式不确定集最坏情形（工时 x1.5）
      sp  : 随机规划场景嵌入（均值 + 1σ 安全边际，σ_t=0.15）
      mc  : 同 post（基础蒙特卡洛对照）
    """
    lp = LAYOUTS[layout]
    n_ws = lp["n_ws"]
    fatigue_mult = lp["fatigue_mult"]

    def eff_time(t):
        if method == "ro":
            return t * 1.5
        if method == "sp":
            return t * 1.15  # 均值 + 1σ 安全边际
        return t

    # 姿态：flex 布局所有工位支持坐+站；刚性布局工位固定姿态（偶数坐、奇数站）
    def posture_ok(ws, posture):
        if lp["posture_flex"]:
            return True
        return (ws % 2) == posture

    cap_t = CAP_TIME * SLACK
    cap_f = CAP_FATIGUE * SLACK
    ws_time = np.zeros(n_ws)
    ws_fat = np.zeros(n_ws)
    plan = []

    # 按产品优先级（价格降序）→ 深度顺序处理；同一零部件精细/粗拆择一
    by_pid = {}
    for c in cands:
        by_pid.setdefault(c["pid"], []).append(c)

    last_ws = {}  # pid -> 已用到的最后工位（前驱位置不降）
    for pid in sorted(by_pid):
        tasks = sorted(by_pid[pid], key=lambda t: t["depth"])
        depths = sorted({t["depth"] for t in tasks})
        for depth in depths:
            chosen = max([t for t in tasks if t["depth"] == depth],
                         key=lambda t: t["value"] / eff_time(t["time"]))
            placed = False
            for ws in range(last_ws.get(pid, 0), n_ws):
                if not posture_ok(ws, chosen["posture"]):
                    continue
                tt = eff_time(chosen["time"])
                ff = chosen["fatigue"] * fatigue_mult
                if ws_time[ws] + tt <= cap_t and ws_fat[ws] + ff <= cap_f:
                    ws_time[ws] += tt
                    ws_fat[ws] += ff
                    plan.append(dict(task=chosen, ws=ws, time=chosen["time"],
                                     fatigue=chosen["fatigue"] * fatigue_mult,
                                     value=chosen["value"], mode=chosen["mode"]))
                    last_ws[pid] = ws
                    placed = True
                    break
            if not placed:
                break  # 该产品更深部件的前驱未就绪，停止该产品
    return plan


# ----------------------------------------------------------------------------
# 3. Stage 2：蒙特卡洛后评估（Normal 工时 / Normal 疲劳 / 伯努利工位故障）
# ----------------------------------------------------------------------------
def simulate_batch(plan, layout, sigma_t=0.15, sigma_e=0.2, p_f=0.05,
                   n_sim=1000, seed=0):
    """对固定方案批量采样扰动，返回 (profits, feasible, fatigue, loss) 数组。

    - 工时：Normal(t, (σ_t·t)²)，截断非负
    - 疲劳：Normal(ε, (σ_ε·ε)²)，截断非负
    - 工位故障：伯努利 p_f，故障则该工位全部任务失效（前驱级联丢失）
    - 可行：扰动后各工位累计工时/疲劳均 ≤ τ_f（容量不超）
    - 利润：存活任务收益 − 处理成本；不可行时按 价值/工时 从低到高丢弃任务
    """
    lp = LAYOUTS[layout]
    n_ws = lp["n_ws"]
    rng = np.random.default_rng(seed)
    n_t = len(plan)

    times_nom = np.array([i["time"] for i in plan])
    fatig_nom = np.array([i["fatigue"] for i in plan])
    values = np.array([i["value"] for i in plan])
    ws_idx = np.array([i["ws"] for i in plan])
    pids = np.array([i["task"]["pid"] for i in plan])
    depths = np.array([i["task"]["depth"] for i in plan])

    # 向量化工时/疲劳扰动（Normal）
    time_mat = np.maximum(0.0, rng.normal(times_nom, sigma_t * times_nom,
                                          size=(n_sim, n_t)))
    fat_mat = np.maximum(0.0, rng.normal(fatig_nom, sigma_e * fatig_nom,
                                         size=(n_sim, n_t)))
    # 工位故障（伯努利）
    fail_mat = rng.random((n_sim, n_ws)) < p_f

    # 前驱链：任务 j 的前驱（同产品深度-1），无前驱为 -1；拓扑序按 (pid, depth)
    pred_of = np.full(n_t, -1, dtype=int)
    for pid in np.unique(pids):
        for j in np.where(pids == pid)[0]:
            preds = np.where((pids == pid) & (depths == depths[j] - 1))[0]
            if preds.size:
                pred_of[j] = preds[0]
    topo = np.lexsort((depths, pids))

    P_nom = values.sum() - times_nom.sum() * UNIT_COST  # 标称利润
    order = np.argsort(values / times_nom)  # 价值/工时升序，恢复丢弃用

    profits = np.empty(n_sim)
    feas = np.empty(n_sim, dtype=bool)
    fat_tot = np.empty(n_sim)
    loss = np.empty(n_sim)

    for s in range(n_sim):
        # 生存任务：工位未故障，且前驱未被级联丢失
        lost = fail_mat[s, ws_idx].copy()
        for j in topo:
            p = pred_of[j]
            if p >= 0 and lost[p]:
                lost[j] = True
        surv = ~lost

        st = np.zeros(n_ws)
        sf = np.zeros(n_ws)
        sj = np.where(surv)[0]
        for j in sj:
            st[ws_idx[j]] += time_mat[s, j]
            sf[ws_idx[j]] += fat_mat[s, j]

        feas[s] = bool(np.all(st <= CAP_TIME) and np.all(sf <= CAP_FATIGUE))

        # 不可行时丢弃价值/工时最低的存活任务，计算利润与疲劳
        revenue = values[sj].sum()
        for j in order:
            if not surv[j]:
                continue
            w = ws_idx[j]
            if st[w] <= CAP_TIME and sf[w] <= CAP_FATIGUE:
                continue
            st[w] -= time_mat[s, j]
            sf[w] -= fat_mat[s, j]
            revenue -= values[j]
        profits[s] = revenue - st.sum() * UNIT_COST
        fat_tot[s] = sf.sum()
        loss[s] = max(0.0, P_nom - profits[s]) if not feas[s] else 0.0
    return profits, feas, fat_tot, loss


def converge_n(prof, feas, n0=800, step=200, tol=0.01):
    """收敛判据：N 从 n0 起每 step 增加，连续两组 E[f_1]/R_feas 相对偏差 < tol。"""
    n = n0
    ep_prev, rf_prev = prof[:n].mean(), feas[:n].mean()
    while n < len(prof):
        ep, rf = prof[:n + step].mean(), feas[:n + step].mean()
        d1 = abs(ep - ep_prev) / abs(ep_prev) if ep_prev else 0.0
        d2 = abs(rf - rf_prev) / abs(rf_prev) if rf_prev else 0.0
        ep_prev, rf_prev = ep, rf
        n += step
        if d1 < tol and d2 < tol:
            break
    return n


# ----------------------------------------------------------------------------
# 4. 主流程
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="cleaned_data.csv")
    ap.add_argument("--products", type=int, default=60)
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    products = load_products(args.csv, args.products, args.seed)
    products = products.sort_values("price", ascending=False).reset_index(drop=True)
    print(f"[1/3] 加载产品: {len(products)} 款手机（按价格降序，高价值优先拆解）")

    all_cands = build_candidates(products)

    rows = []
    # 扰动场景：(名称, 工时CV σ_t, 疲劳CV σ_ε, 故障概率 p_f)
    core_scenarios = [
        ("none",     0.0,  0.0, 0.0),   # 无扰动
        ("fatigue",  0.0,  0.2, 0.0),   # 仅疲劳（σ_ε=0.2）
        ("failure",  0.0,  0.0, 0.10),  # 仅故障（p_f=0.10）
        ("both_low", 0.15, 0.2, 0.05),  # 疲劳+故障 低强度
        ("both_high",0.30, 0.3, 0.15),  # 疲劳+故障 高强度
    ]

    print("[2/3] Stage1 + Stage2 蒙特卡洛模拟中...")
    raw_profit = {}   # (layout, method, scenario) -> profit 样本数组
    raw_feas = {}     # (layout, method, scenario) -> feasible 样本数组
    for layout in LAYOUTS:
        for method in METHODS:
            plan = solve_stage1(all_cands, layout, method)
            if not plan:
                continue
            for sc_name, st_, se, pf in core_scenarios:
                prof, feas, fat, loss = simulate_batch(
                    plan, layout, st_, se, pf, args.sims, args.seed)
                n = converge_n(prof, feas)
                prof, feas, fat, loss = prof[:n], feas[:n], fat[:n], loss[:n]
                raw_profit[(layout, method, sc_name)] = prof
                raw_feas[(layout, method, sc_name)] = feas
                rows.append(dict(
                    layout=layout, layout_name=LAYOUT_NAMES[layout],
                    method=method, scenario=sc_name,
                    exp_profit=prof.mean(), exp_fatigue=fat.mean(),
                    success_rate=feas.mean(), profit_std=prof.std(ddof=1),
                    avg_loss=loss[~feas].mean() if (~feas).any() else 0.0,
                    n_used=n))

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"[3/3] 结果已保存 {args.out} ({len(df)} 行)")

    # --- 汇总打印 ---
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print("\n=== 各布局（后评估法, 疲劳+故障高强度）指标对比 ===")
    sub = df[(df.method == "post") & (df.scenario == "both_high")][
        ["layout_name", "exp_profit", "success_rate", "profit_std", "avg_loss"]]
    print(sub.to_string(index=False))

    # H1: 单因素 ANOVA（后评估法，高扰动场景，4 布局，基于原始样本）
    lnames = list(LAYOUTS)
    pg = [raw_profit[(l, "post", "both_high")] for l in lnames]
    f, p = stats.f_oneway(*pg)
    h, hp = stats.kruskal(*pg)
    print(f"\nH1 利润 ANOVA: F={f:.2f} p={p:.4f} | Kruskal H={h:.2f} p={hp:.4f}")
    # 可行率：卡方检验（2x4 列联表）
    table = [[int(raw_feas[(l, "post", "both_high")].sum()) for l in lnames],
             [int((~raw_feas[(l, "post", "both_high")]).sum()) for l in lnames]]
    chi2, cp, _, _ = stats.chi2_contingency(table)
    print(f"H1 可行率 卡方: chi2={chi2:.2f} p={cp:.4f}")

    # H2: 方法权衡（高扰动场景，4 布局合并）
    print("\n=== H2 方法权衡（疲劳+故障高强度，4 布局平均）===")
    g2 = df[df.scenario == "both_high"]
    for m in METHODS:
        msub = g2[g2.method == m]
        print(f"  方法={m:4s}  E[利润]={msub.exp_profit.mean():9.1f}  "
              f"可行率={msub.success_rate.mean():.3f}  σ={msub.profit_std.mean():8.1f}  "
              f"平均损失={msub.avg_loss.mean():8.1f}")
    post_prof = np.concatenate([raw_profit[(l, "post", "both_high")] for l in lnames])
    ro_prof = np.concatenate([raw_profit[(l, "ro", "both_high")] for l in lnames])
    sp_prof = np.concatenate([raw_profit[(l, "sp", "both_high")] for l in lnames])
    t_pr, p_pr = stats.ttest_ind(post_prof, ro_prof, equal_var=False)
    t_ps, p_ps = stats.ttest_ind(post_prof, sp_prof, equal_var=False)
    print(f"  t检验 post vs RO: t={t_pr:.2f} p={p_pr:.4f}")
    print(f"  t检验 post vs SP: t={t_ps:.2f} p={p_ps:.4f}")


if __name__ == "__main__":
    main()
