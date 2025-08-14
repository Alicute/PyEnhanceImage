// cpp/poisson_nlm.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <mutex>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// -------------------- 工具：哈希键（量化 λ） --------------------
struct Key {
    float a, b;
    bool operator==(const Key& o) const noexcept { return a==o.a && b==o.b; }
};
struct KeyHash {
    std::size_t operator()(Key const& k) const noexcept {
        // 简单混合
        auto ha = std::hash<int>{}(int(k.a*1000));
        auto hb = std::hash<int>{}(int(k.b*1000));
        return ha ^ (hb << 1);
    }
};

// -------------------- d(λx,λy) LUT（L2分布距离） --------------------
static std::unordered_map<Key, double, KeyHash> g_dlut;
static std::mutex g_dlut_mtx;

static double poisson_L2_distance(double lx, double ly) {
    if (lx <= 0.0 && ly <= 0.0) return 0.0;
    double lmax = std::max(lx, ly);
    int Rmax = int(std::ceil(lmax + 6.0*std::sqrt(std::max(lmax, 1e-12))));

    // 递推 PMF： p(0)=e^{-λ}, p(r)=p(r-1)*λ/r
    double px_prev = std::exp(-lx);
    double py_prev = std::exp(-ly);
    double d = (px_prev - py_prev) * (px_prev - py_prev);

    for (int r = 1; r <= Rmax; ++r) {
        px_prev = px_prev * lx / double(r);
        py_prev = py_prev * ly / double(r);
        double diff = px_prev - py_prev;
        d += diff * diff;
    }
    return d;
}

static double dlut_query(double lx, double ly, double lam_quant) {
    // 量化以提升缓存命中
    double qx = std::round(lx / lam_quant) * lam_quant;
    double qy = std::round(ly / lam_quant) * lam_quant;
    Key key{ float(qx), float(qy) };

    {
        std::lock_guard<std::mutex> lock(g_dlut_mtx);
        auto it = g_dlut.find(key);
        if (it != g_dlut.end()) return it->second;
    }
    double val = poisson_L2_distance(qx, qy);
    {
        std::lock_guard<std::mutex> lock(g_dlut_mtx);
        g_dlut.emplace(key, val);
    }
    return val;
}

// 盒均值（用于 λ̂ 的局部均值近似，ksize=2*pr+1）
static inline float box_mean(const float* img, int H, int W, int y0, int x0, int k) {
    int y1 = y0 + k, x1 = x0 + k;
    double s = 0.0;
    for (int y = y0; y < y1; ++y) {
        const float* row = img + y*W;
        for (int x = x0; x < x1; ++x) s += row[x];
    }
    return float(s / double(k*k));
}

// -------------------- 主函数：泊松 NLM 在梯度域 --------------------
py::tuple poisson_nlm_on_gradient_exact_cpp(
    py::array_t<float, py::array::c_style | py::array::forcecast> Gx_p,
    py::array_t<float, py::array::c_style | py::array::forcecast> Gy_p,
    int search_radius, int patch_radius,
    double rho,                // ρ
    double count_target_mean,  // 目标平均 λ（自动尺度）
    double lam_quant,          // λ 量化步长（如 0.02）
    int topk                   // <=0 表示不用 topk
){
    py::buffer_info bx = Gx_p.request();
    py::buffer_info by = Gy_p.request();
    if (bx.ndim != 2 || by.ndim != 2 || bx.shape[0]!=by.shape[0] || bx.shape[1]!=by.shape[1]) {
        throw std::runtime_error("Gx_p/Gy_p must be same 2D shape");
    }
    int H = (int)bx.shape[0], W = (int)bx.shape[1];
    const float* gx_in = (const float*)bx.ptr;
    const float* gy_in = (const float*)by.ptr;

    // 1) 计算 |G'| 与全图均值，确定 count_scale，使均值 λ ≈ count_target_mean
    double sum_mag = 0.0;
    #pragma omp parallel for reduction(+:sum_mag) if(H*W>100000)
    for (int i = 0; i < H*W; ++i) {
        double gx = gx_in[i], gy = gy_in[i];
        sum_mag += std::sqrt(gx*gx + gy*gy);
    }
    double gm = sum_mag / double(H*W);
    double count_scale = (gm > 1e-12) ? (count_target_mean / gm) : 1.0;

    // 2) 计算 λ 图（先不做盒均值），再计算 λ̂（局部均值）
    std::vector<float> lam(H*W), lam_hat(H*W);
    #pragma omp parallel for if(H*W>100000)
    for (int i = 0; i < H*W; ++i) {
        double gx = gx_in[i], gy = gy_in[i];
        double mag = std::sqrt(gx*gx + gy*gy);
        lam[i] = float(std::max(0.0, mag * count_scale));
    }
    int k = 2*patch_radius + 1;
    // 简单盒均值（可替换成积分图/Separable blur）
    #pragma omp parallel for if(H>64)
    for (int y = 0; y < H; ++y) {
        int y0 = std::max(0, y - patch_radius);
        int y1 = std::min(H, y + patch_radius + 1);
        for (int x = 0; x < W; ++x) {
            int x0 = std::max(0, x - patch_radius);
            int x1 = std::min(W, x + patch_radius + 1);
            // 保证边界也能算；与 Python 版本一致性要靠主循环边界裁切
            double s = 0.0;
            int cnt = 0;
            for (int yy = y0; yy < y1; ++yy) {
                const float* row = lam.data() + yy*W;
                for (int xx = x0; xx < x1; ++xx) { s += row[xx]; ++cnt; }
            }
            lam_hat[y*W + x] = float(s / std::max(1, cnt));
        }
    }

    // 3) 输出
    py::array_t<float> Gx({H, W});
    py::array_t<float> Gy({H, W});
    auto bxo = Gx.request(); auto byo = Gy.request();
    float* gx_out = (float*)bxo.ptr;
    float* gy_out = (float*)byo.ptr;

    int pr = patch_radius, sr = search_radius;

    // 4) 主循环：对每个像素做非局部权重加权（严格式(11)(12)）
    #pragma omp parallel for schedule(dynamic, 4) if(H>16)
    for (int y = pr; y < H-pr; ++y) {
        for (int x = pr; x < W-pr; ++x) {
            // x 的 patch 与 λ̂
            int x0p = x - pr, y0p = y - pr;
            float lam_x_bar = box_mean(lam_hat.data(), H, W, y0p, x0p, k);
            double denom = rho * std::max(double(lam_x_bar), 1e-8);

            // 搜索窗
            int sy0 = std::max(pr, y - sr), sy1 = std::min(H - pr, y + sr + 1);
            int sx0 = std::max(pr, x - sr), sx1 = std::min(W - pr, x + sr + 1);

            // 收集候选的 D 与坐标
            std::vector<double> Ds;
            std::vector<std::pair<int,int>> Cs;
            Ds.reserve((sx1-sx0)*(sy1-sy0));
            Cs.reserve(Ds.capacity());

            for (int yy = sy0; yy < sy1; ++yy) {
                for (int xx = sx0; xx < sx1; ++xx) {
                    // Σ_m d(λx_m, λy_m)
                    double D_xy = 0.0;
                    for (int j = 0; j < k; ++j) {
                        const float* row_x = lam_hat.data() + (y0p + j)*W + x0p;
                        const float* row_y = lam_hat.data() + (yy - pr + j)*W + (xx - pr);
                        for (int i = 0; i < k; ++i) {
                            double lx = (double)row_x[i];
                            double ly = (double)row_y[i];
                            D_xy += dlut_query(lx, ly, lam_quant);
                        }
                    }
                    Ds.push_back(D_xy);
                    Cs.emplace_back(yy, xx);
                }
            }

            // 选 topk（可选）
            if (topk > 0 && (int)Ds.size() > topk) {
                std::vector<int> idx(Ds.size());
                std::iota(idx.begin(), idx.end(), 0);
                std::nth_element(idx.begin(), idx.begin()+topk, idx.end(),
                    [&](int a, int b){ return Ds[a] < Ds[b]; });
                idx.resize(topk);

                std::vector<double> Ds2; Ds2.reserve(topk);
                std::vector<std::pair<int,int>> Cs2; Cs2.reserve(topk);
                for (int id : idx) { Ds2.push_back(Ds[id]); Cs2.push_back(Cs[id]); }
                Ds.swap(Ds2); Cs.swap(Cs2);
            }

            // 权重
            std::vector<double> ws; ws.reserve(Ds.size());
            double wsum = 0.0;
            for (double Dxy : Ds) {
                double w = std::exp(- Dxy / denom);
                ws.push_back(w); wsum += w;
            }
            if (wsum <= 0.0) { ws.assign(ws.size(), 1.0); wsum = (double)ws.size(); }

            // 加权平均 G'
            double gxv = 0.0, gyv = 0.0;
            for (size_t i = 0; i < ws.size(); ++i) {
                double w = ws[i] / wsum;
                int yy = Cs[i].first, xx = Cs[i].second;
                gxv += w * (double)gx_in[yy*W + xx];
                gyv += w * (double)gy_in[yy*W + xx];
            }
            gx_out[y*W + x] = float(gxv);
            gy_out[y*W + x] = float(gyv);
        }
    }

    // 边界直接拷回原值
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (y<pr || y>=H-pr || x<pr || x>=W-pr) {
                gx_out[y*W + x] = gx_in[y*W + x];
                gy_out[y*W + x] = gy_in[y*W + x];
            }
        }
    }

    return py::make_tuple(Gx, Gy, count_scale);
}

// 检查OpenMP是否可用
bool is_openmp_available() {
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}

// 获取OpenMP线程数
int get_openmp_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

PYBIND11_MODULE(poisson_nlm_cpp, m) {
    m.doc() = "Strict Poisson NLM on gradient field (pybind11 + OpenMP)";
    m.def("poisson_nlm_on_gradient_exact_cpp", &poisson_nlm_on_gradient_exact_cpp,
          py::arg("Gx_prime"), py::arg("Gy_prime"),
          py::arg("search_radius")=3, py::arg("patch_radius")=1,
          py::arg("rho")=1.5, py::arg("count_target_mean")=30.0,
          py::arg("lam_quant")=0.02, py::arg("topk")=0);
    m.def("is_openmp_available", &is_openmp_available);
    m.def("get_openmp_threads", &get_openmp_threads);
    m.attr("__version__") = "0.1.0";
}

