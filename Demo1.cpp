#include <bits/stdc++.h>
using namespace std;

/*
=========================================================
 智能物流配送系统（实验3 满分版）
 - 路径规划：Dijkstra（最短路）
 - 车辆调度：贪心（需求降序 + 可用车辆优先 + 容量约束）
 - 库存优化：动态规划（多物品多补货方案 + 预算约束最小成本/最大满足）
 - 可选：回溯+分支限界（小规模路线精确优化）
=========================================================
 文件数据（自动生成示例）：
  nodes.txt      : 节点列表（id name）
  edges.txt      : 边（u v w）
  orders.txt     : 订单（orderId nodeId demand serviceTime）
  vehicles.txt   : 车辆（vehId capacity startTime）
  inventory.txt  : 库存（itemName onHand demand unitPenalty  optionCount  [qty cost]...）
=========================================================
*/

static const string kAccount = "1";
static const string kPassword = "123456";

// ---------- 回溯统计（用于展示分支限界效果） ----------
static long long g_dfsCount = 0;
static long long g_pruneCount = 0;

// ---------- 平台清屏/暂停 ----------
static void cls() {
#ifdef _WIN32
    system("cls");
#else
    cout << "\033[2J\033[H";
#endif
}
static void pauseAny() {
#ifdef _WIN32
    system("pause");
#else
    cout << "按回车继续...";
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
    cin.get();
#endif
}

// ---------- 数据结构 ----------
struct Node {
    int id{};
    string name;
};

struct Edge {
    int to{};
    int w{};
};

struct Order {
    int orderId{};
    int nodeId{};
    int demand{};
    int serviceTime{};
    int readyTime{};   // 最早可服务时间
    int dueTime{};     // 最晚完成时间
};


struct Vehicle {
    int vehId{};
    int capacity{};
    int startTime{};     // 可用时间（例如 0 表示立即）
};

struct RoutePlan {
    int vehId{};
    vector<int> orderIds;    // 分配到该车辆的订单ID
    vector<int> visitNodes;  // 实际访问节点序列（含仓库0）
    int totalDemand{};
    int finishTime{};
    int totalDistance{};
};

struct InventoryOption {
    int qty{};
    int cost{};
};

struct InventoryItem {
    string name;
    int onHand{};
    int demand{};
    int unitPenalty{};        // 缺货单位罚金（越大越希望满足）
    vector<InventoryOption> options; // 补货方案：补qty 花cost
};

struct Graph {
    vector<Node> nodes;
    vector<vector<Edge>> adj;
    unordered_map<int,int> id2idx; // nodeId -> index in nodes
};

// ---------- 工具：文件存在 ----------
static bool fileExists(const string& path) {
    ifstream in(path);
    return (bool)in;
}

// ---------- 示例数据生成 ----------
static void writeSampleDataIfMissing() {
    if (!fileExists("nodes.txt")) {
        ofstream out("nodes.txt");
        // 0 号默认仓库
        out << "0 Depot\n";
        out << "1 A\n";
        out << "2 B\n";
        out << "3 C\n";
        out << "4 D\n";
        out << "5 E\n";
    }
    if (!fileExists("edges.txt")) {
        ofstream out("edges.txt");
        // 无向图边：u v w
        out << "0 1 4\n";
        out << "0 2 2\n";
        out << "1 2 1\n";
        out << "1 3 5\n";
        out << "2 3 8\n";
        out << "2 4 10\n";
        out << "3 4 2\n";
        out << "3 5 6\n";
        out << "4 5 3\n";
    }
    if (!fileExists("orders.txt")) {
        ofstream out("orders.txt");
        // orderId nodeId demand serviceTime
        out << "100 1 3 2\n";
        out << "101 2 4 3\n";
        out << "102 3 2 2\n";
        out << "103 4 5 4\n";
        out << "104 5 2 2\n";
    }
    if (!fileExists("vehicles.txt")) {
        ofstream out("vehicles.txt");
        // vehId capacity startTime
        out << "1 8 0\n";
        out << "2 7 0\n";
    }
    if (!fileExists("inventory.txt")) {
        ofstream out("inventory.txt");
        /*
         itemName onHand demand unitPenalty optionCount [qty cost]...
         例如：ItemX 2 8 10 3  (3,8) (5,13) (8,20)
        */
        out << "Widget 2 10 8  3  3 6  6 10  10 16\n";
        out << "Gadget 5 9  6  3  2 4  4 7   6 10\n";
        out << "Bolt   20 30 3  2  10 3  20 5\n";
    }
}

// ---------- 读取数据 ----------
static Graph loadGraph() {
    Graph g;
    ifstream nIn("nodes.txt");
    if (!nIn) throw runtime_error("无法打开 nodes.txt");
    g.nodes.clear();
    int id; string name;
    while (nIn >> id >> name) {
        g.nodes.push_back({id, name});
    }
    g.adj.assign(g.nodes.size(), {});
    for (int i = 0; i < (int)g.nodes.size(); i++) {
        g.id2idx[g.nodes[i].id] = i;
    }
    ifstream eIn("edges.txt");
    if (!eIn) throw runtime_error("无法打开 edges.txt");
    int u, v, w;
    while (eIn >> u >> v >> w) {
        if (!g.id2idx.count(u) || !g.id2idx.count(v)) continue;
        int ui = g.id2idx[u], vi = g.id2idx[v];
        g.adj[ui].push_back({vi, w});
        g.adj[vi].push_back({ui, w});
    }
    return g;
}

static vector<Order> loadOrders() {
    ifstream in("orders.txt");
    if (!in) throw runtime_error("无法打开 orders.txt");
    vector<Order> orders;
    Order o;
	while (in >> o.orderId >> o.nodeId >> o.demand
	          >> o.serviceTime >> o.readyTime >> o.dueTime) {
	    orders.push_back(o);
	}
    return orders;
}

static vector<Vehicle> loadVehicles() {
    ifstream in("vehicles.txt");
    if (!in) throw runtime_error("无法打开 vehicles.txt");
    vector<Vehicle> vs;
    Vehicle v;
    while (in >> v.vehId >> v.capacity >> v.startTime) {
        vs.push_back(v);
    }
    return vs;
}

static vector<InventoryItem> loadInventory() {
    ifstream in("inventory.txt");
    if (!in) throw runtime_error("无法打开 inventory.txt");
    vector<InventoryItem> items;
    while (true) {
        InventoryItem it;
        int optionCount;
        if (!(in >> it.name >> it.onHand >> it.demand >> it.unitPenalty >> optionCount)) break;
        it.options.clear();
        for (int i = 0; i < optionCount; i++) {
            InventoryOption op;
            in >> op.qty >> op.cost;
            it.options.push_back(op);
        }
        items.push_back(it);
    }
    return items;
}

// ---------- 路径规划：Dijkstra 最短路 ----------
static pair<vector<int>, vector<int>> dijkstra(const Graph& g, int srcNodeId) {
    int n = (int)g.nodes.size();
    const int INF = 1e9;
    vector<int> dist(n, INF), parent(n, -1);

    if (!g.id2idx.count(srcNodeId)) throw runtime_error("srcNodeId 不存在");
    int s = g.id2idx.at(srcNodeId);

    using P = pair<int,int>; // dist, idx
    priority_queue<P, vector<P>, greater<P>> pq;
    dist[s] = 0;
    pq.push({0, s});

    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d != dist[u]) continue;
        for (auto e : g.adj[u]) {
            if (dist[e.to] > dist[u] + e.w) {
                dist[e.to] = dist[u] + e.w;
                parent[e.to] = u;
                pq.push({dist[e.to], e.to});
            }
        }
    }
    return {dist, parent};
}

static vector<int> restorePathByIdx(const vector<int>& parent, int srcIdx, int dstIdx) {
    vector<int> path;
    int cur = dstIdx;
    while (cur != -1) {
        path.push_back(cur);
        if (cur == srcIdx) break;
        cur = parent[cur];
    }
    reverse(path.begin(), path.end());
    if (path.empty() || path.front() != srcIdx) return {};
    return path;
}

// ---------- 车辆调度：贪心分配订单到车辆 ----------
// 策略：订单按 demand 降序（First-Fit Decreasing），车辆按“最早可用时间/剩余容量”选择
static vector<RoutePlan> scheduleVehiclesWithTimeWindow(
    const vector<Order>& orders,
    const vector<Vehicle>& vehicles
) {
    // 按 dueTime（最紧急）优先
    vector<Order> ord = orders;
    sort(ord.begin(), ord.end(), [](const Order& a, const Order& b) {
        return a.dueTime < b.dueTime;
    });

    struct VehState {
        Vehicle v;
        int remainingCap;
        int currentTime;
        vector<int> assignedOrderIds;
        int totalDemand;
    };

    vector<VehState> states;
    for (auto& v : vehicles) {
        states.push_back({v, v.capacity, v.startTime, {}, 0});
    }

    vector<int> unassigned;

    for (auto& o : ord) {
        int best = -1;
        int bestFinish = INT_MAX;

        for (int i = 0; i < (int)states.size(); i++) {
            if (states[i].remainingCap < o.demand) continue;

            int arrival = max(states[i].currentTime, o.readyTime);
            int finish = arrival + o.serviceTime;

            if (finish <= o.dueTime) {
                if (finish < bestFinish) {
                    bestFinish = finish;
                    best = i;
                }
            }
        }

        if (best == -1) {
            unassigned.push_back(o.orderId);
            continue;
        }

        states[best].assignedOrderIds.push_back(o.orderId);
        states[best].remainingCap -= o.demand;
        states[best].currentTime = bestFinish;
        states[best].totalDemand += o.demand;
    }

    vector<RoutePlan> plans;
    for (auto& st : states) {
        RoutePlan rp;
        rp.vehId = st.v.vehId;
        rp.orderIds = st.assignedOrderIds;
        rp.totalDemand = st.totalDemand;
        rp.finishTime = st.currentTime;
        plans.push_back(rp);
    }

    if (!unassigned.empty()) {
        cout << "\n? 未分配订单（时间窗/容量限制）：";
        for (int id : unassigned) cout << id << " ";
        cout << "\n";
    }

    return plans;
}

// ---------- 路线生成：贪心最近邻（基于最短路距离矩阵） ----------
static vector<vector<int>> allPairsShortestPathDist(const Graph& g) {
    int n = (int)g.nodes.size();
    vector<vector<int>> dist(n, vector<int>(n, (int)1e9));
    for (int i = 0; i < n; i++) {
        auto [d, p] = dijkstra(g, g.nodes[i].id);
        for (int j = 0; j < n; j++) dist[i][j] = d[j];
    }
    return dist;
}

static vector<int> buildRouteNearestNeighbor(
    const Graph& g,
    const vector<vector<int>>& distMat,
    int depotId,
    const vector<int>& customerNodeIds
) {
    if (!g.id2idx.count(depotId)) throw runtime_error("depotId 不存在");
    int depotIdx = g.id2idx.at(depotId);

    vector<int> targets;
    for (int nid : customerNodeIds) {
        if (g.id2idx.count(nid)) targets.push_back(g.id2idx.at(nid));
    }
    sort(targets.begin(), targets.end());
    targets.erase(unique(targets.begin(), targets.end()), targets.end());

    vector<int> routeIdx;
    routeIdx.push_back(depotIdx);

    vector<int> unvisited = targets;
    int cur = depotIdx;

    while (!unvisited.empty()) {
        int bestPos = 0;
        for (int i = 1; i < (int)unvisited.size(); i++) {
            if (distMat[cur][unvisited[i]] < distMat[cur][unvisited[bestPos]]) {
                bestPos = i;
            }
        }
        int nxt = unvisited[bestPos];
        routeIdx.push_back(nxt);
        cur = nxt;
        unvisited.erase(unvisited.begin() + bestPos);
    }

    // 回仓库
    routeIdx.push_back(depotIdx);

    // 转成 nodeId
    vector<int> routeNodeId;
    for (int idx : routeIdx) routeNodeId.push_back(g.nodes[idx].id);
    return routeNodeId;
}

// ---------- 回溯 + 分支限界：小规模精确路线（可选） ----------
// 对每辆车的客户点 <= 10 时，尝试找更优的闭环路径（TSP 变体）
// 使用 distMat，进行回溯搜索 + 当前最短上界剪枝
static void tspBacktrackBB(
    const vector<vector<int>>& distMat,
    int depotIdx,
    const vector<int>& customerIdx,
    vector<int>& bestPath,
    int& bestCost
) {
    int m = (int)customerIdx.size();
    vector<int> path;
    vector<int> used(m, 0);

    // 预计算最小边（用于下界估计）
    int globalMinEdge = INT_MAX;
    for (int i = 0; i < (int)distMat.size(); i++)
        for (int j = 0; j < (int)distMat.size(); j++)
            if (i != j && distMat[i][j] < globalMinEdge)
                globalMinEdge = distMat[i][j];

    function<void(int,int,int)> dfs = [&](int depth, int lastIdx, int costSoFar) {
        g_dfsCount++;

        // ===== 下界估计（分支限界核心）=====
        int remaining = m - depth;
        int lowerBound = costSoFar + remaining * globalMinEdge;
        if (lowerBound >= bestCost) {
            g_pruneCount++;
            return;
        }

        if (depth == m) {
            int total = costSoFar + distMat[lastIdx][depotIdx];
            if (total < bestCost) {
                bestCost = total;
                bestPath = path;
            }
            return;
        }

        for (int i = 0; i < m; i++) {
            if (used[i]) continue;
            int nxt = customerIdx[i];
            int add = distMat[lastIdx][nxt];
            if (add >= (int)1e9) continue;

            used[i] = 1;
            path.push_back(nxt);
            dfs(depth + 1, nxt, costSoFar + add);
            path.pop_back();
            used[i] = 0;
        }
    };

    dfs(0, depotIdx, 0);
}


static vector<int> buildRouteGreedyOrExact(
    const Graph& g,
    const vector<vector<int>>& distMat,
    int depotId,
    const vector<int>& customerNodeIds
) {
    // ---------- 1. 贪心上界 ----------
    vector<int> greedy = buildRouteNearestNeighbor(g, distMat, depotId, customerNodeIds);

    auto calcCost = [&](const vector<int>& routeNodeId) {
        int cost = 0;
        for (int i = 1; i < (int)routeNodeId.size(); i++) {
            int a = g.id2idx.at(routeNodeId[i-1]);
            int b = g.id2idx.at(routeNodeId[i]);
            cost += distMat[a][b];
        }
        return cost;
    };

    int greedyCost = calcCost(greedy);

    // ---------- 2. 客户点去重 ----------
    unordered_set<int> s;
    for (int nid : customerNodeIds) s.insert(nid);
    if ((int)s.size() > 10) {
        cout << "[提示] 客户点数量 > 10，自动使用贪心路线。\n";
        return greedy;
    }

    // ---------- 3. 精确搜索 ----------
    vector<int> customerIdx;
    for (int nid : s) customerIdx.push_back(g.id2idx.at(nid));

    int depotIdx = g.id2idx.at(depotId);
    int bestCost = greedyCost;
    vector<int> bestPermIdx;

    g_dfsCount = g_pruneCount = 0;
    tspBacktrackBB(distMat, depotIdx, customerIdx, bestPermIdx, bestCost);

    cout << "------ 路线优化对比（回溯 + 分支限界）------\n";
    cout << "贪心路线距离: " << greedyCost << "\n";
    cout << "最优路线距离: " << bestCost << "\n";
    cout << "搜索节点数: " << g_dfsCount << "\n";
    cout << "剪枝次数: " << g_pruneCount << "\n";

    if (bestCost < greedyCost) {
        double improve = 100.0 * (greedyCost - bestCost) / greedyCost;
        cout << "优化率: " << fixed << setprecision(2) << improve << "%\n";
    } else {
        cout << "未优于贪心（贪心已是最优或接近最优）\n";
    }
    cout << "------------------------------------------\n";

    // ---------- 4. 构造最优路线 ----------
    vector<int> bestRouteNodeId;
    bestRouteNodeId.push_back(depotId);
    for (int idx : bestPermIdx) bestRouteNodeId.push_back(g.nodes[idx].id);
    bestRouteNodeId.push_back(depotId);

    return bestRouteNodeId;
}

static void twoOptImprove(
    vector<int>& route,
    const Graph& g,
    const vector<vector<int>>& distMat
) {
    bool improved = true;
    int n = route.size();

    auto dist = [&](int a, int b) {
        return distMat[g.id2idx.at(a)][g.id2idx.at(b)];
    };

    while (improved) {
        improved = false;
        for (int i = 1; i < n - 2; i++) {
            for (int k = i + 1; k < n - 1; k++) {
                int delta =
                    dist(route[i - 1], route[k]) +
                    dist(route[i], route[k + 1]) -
                    dist(route[i - 1], route[i]) -
                    dist(route[k], route[k + 1]);

                if (delta < 0) {
                    reverse(route.begin() + i, route.begin() + k + 1);
                    improved = true;
                }
            }
        }
    }
}



// ---------- 路线距离 ----------
static int routeDistance(const Graph& g, const vector<vector<int>>& distMat, const vector<int>& routeNodeIds) {
    int sum = 0;
    for (int i = 1; i < (int)routeNodeIds.size(); i++) {
        int a = g.id2idx.at(routeNodeIds[i-1]);
        int b = g.id2idx.at(routeNodeIds[i]);
        sum += distMat[a][b];
    }
    return sum;
}



// ---------- 库存优化：动态规划 ----------
// 目标：在预算 B 内，为每个物品选择一个补货方案（或不补），使“缺货罚金最小”（等价于满足最大）
// 缺货量 = max(0, demand - (onHand + replenishQty))
// 成本 = sum(option.cost)
// 罚金 = sum(缺货量 * unitPenalty)
// DP[itemIndex][budget] = 最小罚金
static void optimizeInventoryDP(const vector<InventoryItem>& items, int budget) {
    int n = (int)items.size();
    const long long INF = (1LL<<60);

    vector<vector<long long>> dp(n+1, vector<long long>(budget+1, INF));
    vector<vector<int>> choice(n+1, vector<int>(budget+1, -1)); // 记录选哪个 option（-1表示不补）
    dp[0][0] = 0;

    auto penalty = [&](const InventoryItem& it, int addQty) -> long long {
        int have = it.onHand + addQty;
        int shortage = max(0, it.demand - have);
        return 1LL * shortage * it.unitPenalty;
    };

    for (int i = 1; i <= n; i++) {
        const auto& it = items[i-1];
        for (int b = 0; b <= budget; b++) {
            // 不补货
            if (dp[i-1][b] < INF) {
                long long val = dp[i-1][b] + penalty(it, 0);
                if (val < dp[i][b]) {
                    dp[i][b] = val;
                    choice[i][b] = -1;
                }
            }
            // 选一个补货方案
            for (int k = 0; k < (int)it.options.size(); k++) {
                int cost = it.options[k].cost;
                int qty  = it.options[k].qty;
                if (b >= cost && dp[i-1][b-cost] < INF) {
                    long long val = dp[i-1][b-cost] + penalty(it, qty);
                    if (val < dp[i][b]) {
                        dp[i][b] = val;
                        choice[i][b] = k;
                    }
                }
            }
        }
    }

    // 找到最优预算点
    int bestB = 0;
    for (int b = 1; b <= budget; b++) {
        if (dp[n][b] < dp[n][bestB]) bestB = b;
    }

    // 回溯方案
    vector<int> picked(n, -1);
    int b = bestB;
    for (int i = n; i >= 1; i--) {
        int c = choice[i][b];
        picked[i-1] = c;
        if (c != -1) b -= items[i-1].options[c].cost;
    }

    // 输出
    cout << "\n========== 库存优化（动态规划DP） ==========\n";
    cout << "预算上限: " << budget << "，最优使用预算: " << bestB << "\n";
    cout << "最小缺货罚金: " << dp[n][bestB] << "\n\n";

    cout << left << setw(12) << "物品"
         << setw(8)  << "现有"
         << setw(8)  << "需求"
         << setw(10) << "补货"
         << setw(8)  << "成本"
         << setw(10) << "缺货"
         << setw(10) << "罚金" << "\n";

    long long totalCost = 0, totalPenalty = 0;
    for (int i = 0; i < n; i++) {
        int addQty = 0, cost = 0;
        if (picked[i] != -1) {
            addQty = items[i].options[picked[i]].qty;
            cost = items[i].options[picked[i]].cost;
        }
        int have = items[i].onHand + addQty;
        int shortage = max(0, items[i].demand - have);
        long long pen = 1LL * shortage * items[i].unitPenalty;
        totalCost += cost;
        totalPenalty += pen;

        cout << left << setw(12) << items[i].name
             << setw(8)  << items[i].onHand
             << setw(8)  << items[i].demand
             << setw(10) << addQty
             << setw(8)  << cost
             << setw(10) << shortage
             << setw(10) << pen << "\n";
    }
    cout << "总成本: " << totalCost << "，总罚金: " << totalPenalty << "\n";
    cout << "===========================================\n";
}

// ---------- 打印基础数据 ----------
static void showData(const Graph& g, const vector<Order>& orders, const vector<Vehicle>& vehicles, const vector<InventoryItem>& inv) {
    cout << "\n===== 节点 Nodes =====\n";
    for (auto &n : g.nodes) cout << "Node " << n.id << " : " << n.name << "\n";

    cout << "\n===== 订单 Orders =====\n";
    for (auto &o : orders) {
        cout << "Order " << o.orderId << " -> Node " << o.nodeId
             << " demand=" << o.demand << " serviceTime=" << o.serviceTime << "\n";
    }

    cout << "\n===== 车辆 Vehicles =====\n";
    for (auto &v : vehicles) {
        cout << "Vehicle " << v.vehId << " cap=" << v.capacity << " start=" << v.startTime << "\n";
    }

    cout << "\n===== 库存 Inventory =====\n";
    for (auto &it : inv) {
        cout << it.name << " onHand=" << it.onHand << " demand=" << it.demand
             << " penalty=" << it.unitPenalty << " options:";
        for (auto &op : it.options) cout << " (" << op.qty << "," << op.cost << ")";
        cout << "\n";
    }
}

// ---------- 主功能：规划配送（调度+路径） ----------
static void planDelivery(
    const Graph& g,
    const vector<Order>& orders,
    const vector<Vehicle>& vehicles,
    bool useExactIfSmall
) {
    // 1) 带时间窗的贪心调度
    auto plans = scheduleVehiclesWithTimeWindow(orders, vehicles);

    // 2) 最短路距离矩阵
    auto distMat = allPairsShortestPathDist(g);

    unordered_map<int, Order> mp;
    for (auto &o : orders) mp[o.orderId] = o;

    cout << "\n========== 车辆调度 + 路线规划结果 ==========\n";

    for (auto &rp : plans) {
        if (rp.orderIds.empty()) {
            cout << "\n[车辆 " << rp.vehId << "] 无分配订单。\n";
            continue;
        }

        vector<int> customerNodes;
        for (int oid : rp.orderIds)
            customerNodes.push_back(mp[oid].nodeId);

        // 3) 初始路线
        vector<int> route;
        bool smallScale = (unordered_set<int>(
            customerNodes.begin(), customerNodes.end()).size() <= 10);

        if (useExactIfSmall && smallScale) {
            route = buildRouteGreedyOrExact(g, distMat, 0, customerNodes);
        } else {
            route = buildRouteNearestNeighbor(g, distMat, 0, customerNodes);
        }

        int before = routeDistance(g, distMat, route);

        // 4) 仅在大规模下启用 2-opt
        if (!smallScale) {
            twoOptImprove(route, g, distMat);
        }

        int after = routeDistance(g, distMat, route);

        rp.visitNodes = route;
        rp.totalDistance = after;
        rp.finishTime += after; // 距离≈时间（简化模型）

        cout << "\n[车辆 " << rp.vehId << "] 分配订单: ";
        for (int oid : rp.orderIds) cout << oid << " ";

        cout << "\n路线距离: " << before;
        if (!smallScale)
            cout << " → " << after
                 << "（2-opt 优化 "
                 << fixed << setprecision(2)
                 << 100.0 * (before - after) / before << "%）";

        cout << "\n预计完成时间: " << rp.finishTime << "\n路线: ";

        for (int nid : route) {
            cout << g.nodes[g.id2idx.at(nid)].name << "(" << nid << ")";
            if (&nid != &route.back()) cout << " -> ";
        }
        cout << "\n";
    }

    cout << "=======================================\n";
}

// ---------- 路径查询：Dijkstra 输出某两点最短路径 ----------
static void queryShortestPath(const Graph& g) {
    int sId, tId;
    cout << "输入起点nodeId：";
    cin >> sId;
    cout << "输入终点nodeId：";
    cin >> tId;
    if (!g.id2idx.count(sId) || !g.id2idx.count(tId)) {
        cout << "节点不存在。\n";
        return;
    }
    auto [dist, parent] = dijkstra(g, sId);
    int sIdx = g.id2idx.at(sId);
    int tIdx = g.id2idx.at(tId);
    if (dist[tIdx] >= (int)1e9) {
        cout << "不可达。\n";
        return;
    }
    auto pathIdx = restorePathByIdx(parent, sIdx, tIdx);
    cout << "最短距离 = " << dist[tIdx] << "\n路径：";
    for (int i = 0; i < (int)pathIdx.size(); i++) {
        cout << g.nodes[pathIdx[i]].name << "(" << g.nodes[pathIdx[i]].id << ")";
        if (i + 1 < (int)pathIdx.size()) cout << " -> ";
    }
    cout << "\n";
}

// ---------- 登录 ----------
static bool login(int remain) {
    string acc, pwd;

    cout << "=========================================\n";
    cout << "   智能物流配送系统（实验3 演示版）\n";
    cout << "=========================================\n";
    cout << "【登录提示】\n";
    cout << " - 默认演示账号：1\n";
    cout << " - 默认演示密码：123456\n";
    cout << " - 剩余登录次数：" << remain << "\n";
    cout << "-----------------------------------------\n";

    cout << "请输入账号：";
    cin >> acc;
    cout << "请输入密码：";
    cin >> pwd;

    if (acc == kAccount && pwd == kPassword) {
        cout << "\n? 登录成功！欢迎进入系统。\n";
        cout << "系统将自动加载示例数据用于算法演示。\n";
        return true;
    } else {
        cout << "\n? 登录失败：账号或密码错误。\n";
        if (remain > 1) {
            cout << "请重新输入（剩余 " << remain - 1 << " 次机会）。\n";
        }
        return false;
    }
}

static void demoRouteOptimization(const Graph& g, const vector<Order>& orders) {
    auto distMat = allPairsShortestPathDist(g);

    vector<int> customerNodes;
    for (auto &o : orders) customerNodes.push_back(o.nodeId);

    cout << "\n=== 路线优化对比演示 ===\n";

    auto greedy = buildRouteNearestNeighbor(g, distMat, 0, customerNodes);
    int d1 = routeDistance(g, distMat, greedy);

    auto opt = greedy;
    twoOptImprove(opt, g, distMat);
    int d2 = routeDistance(g, distMat, opt);

    cout << "贪心路线距离: " << d1 << "\n";
    cout << "2-opt 优化后距离: " << d2 << "\n";
    cout << "优化率: "
         << fixed << setprecision(2)
         << 100.0 * (d1 - d2) / d1 << "%\n\n";

    cout << "优化后路线：\n";
    for (int nid : opt) {
        cout << g.nodes[g.id2idx.at(nid)].name << "(" << nid << ")";
        if (&nid != &opt.back()) cout << " -> ";
    }
    cout << "\n";
}



// ---------- 菜单 ----------
static void menuLoop() {
    // 自动生成示例数据
    writeSampleDataIfMissing();

    // 加载数据
    Graph g = loadGraph();
    vector<Order> orders = loadOrders();
    vector<Vehicle> vehicles = loadVehicles();
    vector<InventoryItem> inv = loadInventory();

    while (true) {
	cout << "\n=============================\n";
	cout << "  智能物流配送系统（实验3）\n";
	cout << "=============================\n";
	cout << "1. 查看基础数据\n";
	cout << "2. 路径规划：两点最短路径（Dijkstra）\n";
	cout << "3. 配送规划（时间窗调度 + 路线优化）\n";
	cout << "4. 路线优化对比演示（贪心 vs 2-opt）\n";
	cout << "5. 库存优化（动态规划 DP）\n";
	cout << "6. 退出系统\n";
	cout << "请输入功能编号：";


        int op;
        cin >> op;

        cls();
        if (op == 1) {
            showData(g, orders, vehicles, inv);
            pauseAny();
            cls();
        } else if (op == 2) {
            queryShortestPath(g);
            pauseAny();
            cls();
        } else if (op == 3) {
		    cout << "是否启用小规模精确路线优化（回溯+分支限界，<=10客户）？\n";
		    cout << "1 = 是（精确）   0 = 否（启发式）\n";
		    int yes;
		    cin >> yes;
		
		    cout << "\n【说明】系统将自动：\n";
		    cout << "- 使用带时间窗的贪心算法进行车辆调度\n";
		    cout << "- 小规模使用回溯，大规模使用 2-opt 优化\n\n";
		
		    planDelivery(g, orders, vehicles, yes == 1);
		    pauseAny();
		    cls();
		} else if (op == 4) {
		    demoRouteOptimization(g, orders);
		    pauseAny();
		    cls();
		} else if (op == 5) {
            cout << "输入库存补货总预算（建议 5~30）：";
            int B; cin >> B;
            optimizeInventoryDP(inv, B);
            pauseAny();
            cls();
        } else if (op == 6) {
            cout << "退出。\n";
            break;
        } else {
            cout << "无效输入。\n";
            pauseAny();
            cls();
        }
    }
}

int readIntInRange(const string& tip, int l, int r) {
    int x;
    while (true) {
        cout << tip;
        if (cin >> x && x >= l && x <= r) return x;
        cout << "输入非法，请输入 [" << l << "," << r << "] 之间的整数。\n";
        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
    }
}


int main() {


    // 登录三次机会
    for (int i = 0; i < 3; i++) {
	    if (login(3 - i)) {
	        cls();
	        menuLoop();
	        return 0;
	    }
	}
	cout << "\n? 超过最大登录次数，系统已退出。\n";
	
    return 0;
}
