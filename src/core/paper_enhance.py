import numpy as np
import cv2
from math import ceil, sqrt
from functools import lru_cache
from scipy.stats import poisson
from skimage.restoration import denoise_nl_means

# -------- 规范化到 [0,1]（浮点），并返回可逆上下文 --------
def normalize_to_unit(img, mode="percentile", p_lo=0.5, p_hi=99.5,
                      wl=None, ww=None):
    img = img.astype(np.float32)
    if mode == "window" and wl is not None and ww is not None:
        vmin = wl - ww/2.0
        vmax = wl + ww/2.0
    else:
        vmin, vmax = np.percentile(img, [p_lo, p_hi])
        if vmax <= vmin:
            vmax = img.max() if img.max() > vmin else (vmin + 1.0)
    out = (img - vmin) / (vmax - vmin)
    out = np.clip(out, 0.0, 1.0)
    ctx = {"vmin": float(vmin), "vmax": float(vmax)}
    return out, ctx

def denormalize_from_unit(img_unit, ctx, out_dtype=np.uint16):
    vmin, vmax = ctx["vmin"], ctx["vmax"]
    out = np.clip(img_unit, 0.0, 1.0) * (vmax - vmin) + vmin
    if out_dtype == np.uint16:
        out = np.clip(out, 0, 65535)
    return out.astype(out_dtype)

# -------- 基础算子（全在浮点域） --------
def grad2d(img):
    gx = 0.5 * (np.roll(img, -1, axis=1) - np.roll(img, 1, axis=1))
    gy = 0.5 * (np.roll(img, -1, axis=0) - np.roll(img, 1, axis=0))
    return gx, gy

def div2d(px, py):
    dx = 0.5 * (np.roll(px, -1, axis=1) - np.roll(px, 1, axis=1))
    dy = 0.5 * (np.roll(py, -1, axis=0) - np.roll(py, 1, axis=0))
    return dx + dy

def laplace(img):
    return (np.roll(img, -1, 0) + np.roll(img, 1, 0) +
            np.roll(img, -1, 1) + np.roll(img, 1, 1) - 4*img)

def local_variance(img, ksize=5):
    if ksize % 2 == 0: ksize += 1
    mean = cv2.blur(img, (ksize, ksize), borderType=cv2.BORDER_REFLECT)
    mean2 = cv2.blur(img*img, (ksize, ksize), borderType=cv2.BORDER_REFLECT)
    return np.clip(mean2 - mean*mean, 0, None)

# -------- Step 1：梯度场 + 局部方差自适应增强（严格按式(5)(6)） --------
def adaptive_gradient_enhance_unit(R_unit,
                                   epsilon_unit=2.3/(255.0*255.0),  # ε 映射到 [0,1] 量纲
                                   mu=10.0, ksize_var=5):
    gx, gy = grad2d(R_unit)
    sigma2 = local_variance(R_unit, ksize=ksize_var)  # 局部方差
    sigma = np.sqrt(sigma2 + 1e-12)
    # C：用 sigma 的 90%分位（与论文用梯度分位同阶）
    C = np.percentile(sigma, 90.0) + 1e-12
    k = (1.0 + mu) / (1.0 + (sigma / C)**2)          # 式(5)
    mask = (sigma2 > epsilon_unit).astype(np.float32) # 式(6) 的阈值在 [0,1] 量纲
    Gx_prime = k * gx * mask
    Gy_prime = k * gy * mask
    return Gx_prime, Gy_prime

# -------- Step 2：泊松分布 NLM（严格按式(7)(8)(9)(10)(11)(12)） --------
@lru_cache(maxsize=200000)
def poisson_L2_distance(lx_r, ly_r):
    lx = float(lx_r); ly = float(ly_r)
    lmax = max(lx, ly)
    if lmax <= 0:  # 两个都是0时距离为0
        return 0.0
    Rmax = int(ceil(lmax + 6.0*sqrt(lmax)))
    r = np.arange(0, Rmax + 1)
    px = poisson.pmf(r, lx)
    py = poisson.pmf(r, ly)
    return float(np.sum((px - py)**2))

def estimate_lambda_map(count_img, ksize=3):
    if ksize % 2 == 0: ksize += 1
    lam = cv2.blur(count_img, (ksize, ksize), borderType=cv2.BORDER_REFLECT)
    return np.clip(lam, 1e-8, None)

def poisson_nlm_on_gradient_exact(Gx_prime, Gy_prime,
                                  search_radius=5, patch_radius=1,
                                  rho=1.5,
                                  count_target_mean=30.0,  # 目标平均 λ（数十更稳）
                                  lam_quant=0.02,          # λ 量化步长→缓存命中
                                  topk=None,
                                  progress_callback=None):
    import time
    H, W = Gx_prime.shape
    total_pixels = H * W
    print(f"      🔧 [poisson_nlm] 处理 {H}x{W} 图像，总像素: {total_pixels:,}")

    Gmag = np.sqrt(Gx_prime*Gx_prime + Gy_prime*Gy_prime)

    # 自动把 |G'| 线性映射到“计数”域：使得全图平均 λ ≈ count_target_mean
    print(f"      📊 [poisson_nlm] 计算梯度幅值和λ映射...")
    gm = float(np.mean(Gmag)) + 1e-12
    count_scale = count_target_mean / gm
    counts = np.clip(Gmag * count_scale, 0.0, None).astype(np.float32)

    lam_map = estimate_lambda_map(counts, ksize=3)
    print(f"      ✅ [poisson_nlm] λ映射完成，平均λ: {np.mean(lam_map):.2f}")

    pr = patch_radius
    sr = search_radius
    Gx = np.zeros_like(Gx_prime, dtype=np.float32)
    Gy = np.zeros_like(Gy_prime, dtype=np.float32)

    print(f"      🔄 [poisson_nlm] 开始双重循环处理，预计耗时较长...")
    print(f"         参数: search_radius={sr}, patch_radius={pr}, topk={topk}")

    loop_start = time.time()
    processed_pixels = 0
    last_progress_time = time.time()

    for y in range(H):
        # 每处理一行输出进度
        if y % max(1, H // 20) == 0:  # 每5%输出一次
            current_time = time.time()
            elapsed = current_time - loop_start
            progress_pct = (y / H) * 100
            if y > 0:
                eta = elapsed * (H - y) / y
                print(f"         进度: {progress_pct:.1f}% ({y}/{H}行), 已耗时: {elapsed:.1f}s, 预计剩余: {eta:.1f}s")

            # 更新外部进度回调
            if progress_callback:
                # 泊松NLM在整个算法中占0.4-0.8的进度
                nlm_progress = 0.4 + (y / H) * 0.4
                progress_callback(nlm_progress)

        for x in range(W):
            processed_pixels += 1
            # 取 x 的 patch（用于 Σ_m）
            y0p, y1p = y-pr, y+pr+1
            x0p, x1p = x-pr, x+pr+1
            if y0p < 0 or x0p < 0 or y1p > H or x1p > W:
                Gx[y, x] = Gx_prime[y, x]; Gy[y, x] = Gy_prime[y, x]; continue
            lam_patch_x = lam_map[y0p:y1p, x0p:x1p]
            lam_x_bar = float(np.mean(lam_patch_x))

            # 搜索窗（保证候选也有完整 patch）
            sy0, sy1 = max(pr, y-sr), min(H-pr, y+sr+1)
            sx0, sx1 = max(pr, x-sr), min(W-pr, x+sr+1)

            ds = []; coords = []
            for yy in range(sy0, sy1):
                for xx in range(sx0, sx1):
                    lam_patch_y = lam_map[yy-pr:yy+pr+1, xx-pr:xx+pr+1]

                    # 严格：Σ_m d(λx_m, λy_m)；d 用式(9)(10) 的 L2 分布距离（离散求和）
                    D_xy = 0.0
                    for j in range(lam_patch_x.shape[0]):
                        for i in range(lam_patch_x.shape[1]):
                            lx = lam_patch_x[j, i]; ly = lam_patch_y[j, i]
                            lx_r = round(float(lx), 2) if lam_quant is None else round(float(lx)/lam_quant)*lam_quant
                            ly_r = round(float(ly), 2) if lam_quant is None else round(float(ly)/lam_quant)*lam_quant
                            D_xy += poisson_L2_distance(lx_r, ly_r)
                    ds.append(D_xy); coords.append((yy, xx))

            ds = np.array(ds, dtype=np.float32)
            if topk is not None and len(ds) > topk:
                idx = np.argpartition(ds, topk)[:topk]
                ds = ds[idx]; coords = [coords[i] for i in idx]

            denom = rho * max(lam_x_bar, 1e-8)
            ws = np.exp(- ds / denom).astype(np.float32)
            if ws.sum() == 0: ws = np.ones_like(ws)
            ws /= ws.sum()

            gx_val = 0.0; gy_val = 0.0
            for w, (yy, xx) in zip(ws, coords):
                gx_val += w * Gx_prime[yy, xx]
                gy_val += w * Gy_prime[yy, xx]
            Gx[y, x] = gx_val; Gy[y, x] = gy_val

    loop_time = time.time() - loop_start
    print(f"      ✅ [poisson_nlm] 双重循环完成，耗时: {loop_time:.2f}s，处理了 {processed_pixels:,} 像素")
    print(f"         平均每像素耗时: {loop_time/processed_pixels*1000:.3f}ms")

    return Gx, Gy

# -------- 快速NLM实现（基于skimage + 分块处理） --------
def fast_nlm_on_gradient(Gx_prime, Gy_prime,
                        patch_size=5, patch_distance=6,
                        h=0.1, fast_mode=True,
                        block_size=1024, overlap=16,
                        progress_callback=None):
    """
    使用skimage的快速NLM对梯度场进行去噪

    Args:
        Gx_prime, Gy_prime: 增强后的梯度场
        patch_size: patch大小 (对应原来的patch_radius*2+1)
        patch_distance: 搜索距离 (对应原来的search_radius*2+1)
        h: 滤波强度
        fast_mode: 是否使用快速模式
        block_size: 分块大小
        overlap: 重叠区域大小
        progress_callback: 进度回调

    Returns:
        Gx, Gy: 去噪后的梯度场
    """
    import time
    H, W = Gx_prime.shape
    total_pixels = H * W

    print(f"      🚀 [fast_nlm] 使用skimage快速NLM处理 {H}x{W} 图像")
    print(f"         参数: patch_size={patch_size}, patch_distance={patch_distance}, h={h:.3f}")
    print(f"         分块: {block_size}x{block_size}, overlap={overlap}")

    # 梯度预处理：确保数据范围适合NLM
    print(f"      📊 [fast_nlm] 梯度预处理...")
    gx_min, gx_max = Gx_prime.min(), Gx_prime.max()
    gy_min, gy_max = Gy_prime.min(), Gy_prime.max()
    print(f"         原始梯度范围: Gx[{gx_min:.3f}, {gx_max:.3f}], Gy[{gy_min:.3f}, {gy_max:.3f}]")

    # 将梯度归一化到[0,1]范围，这对NLM很重要
    gx_range = gx_max - gx_min
    gy_range = gy_max - gy_min

    if gx_range > 0:
        Gx_norm = (Gx_prime - gx_min) / gx_range
    else:
        Gx_norm = Gx_prime.copy()

    if gy_range > 0:
        Gy_norm = (Gy_prime - gy_min) / gy_range
    else:
        Gy_norm = Gy_prime.copy()

    print(f"         归一化后范围: Gx[{Gx_norm.min():.3f}, {Gx_norm.max():.3f}], Gy[{Gy_norm.min():.3f}, {Gy_norm.max():.3f}]")

    # 如果图像较小，直接处理
    if total_pixels < 1000000:  # 1M像素以下直接处理
        print(f"      📊 [fast_nlm] 小图像直接处理...")
        start_time = time.time()

        Gx_denoised = denoise_nl_means(Gx_norm, patch_size=patch_size,
                                      patch_distance=patch_distance, h=h,
                                      fast_mode=fast_mode)
        Gy_denoised = denoise_nl_means(Gy_norm, patch_size=patch_size,
                                      patch_distance=patch_distance, h=h,
                                      fast_mode=fast_mode)

        # 反归一化回原始范围
        Gx = Gx_denoised * gx_range + gx_min if gx_range > 0 else Gx_denoised
        Gy = Gy_denoised * gy_range + gy_min if gy_range > 0 else Gy_denoised

        elapsed = time.time() - start_time
        print(f"      ✅ [fast_nlm] 直接处理完成，耗时: {elapsed:.2f}s")
        return Gx, Gy

    # 大图像分块处理
    print(f"      📦 [fast_nlm] 大图像分块处理...")
    start_time = time.time()

    Gx_denoised = np.zeros_like(Gx_norm, dtype=np.float32)
    Gy_denoised = np.zeros_like(Gy_norm, dtype=np.float32)

    # 计算分块数量
    blocks_y = (H + block_size - 1) // block_size
    blocks_x = (W + block_size - 1) // block_size
    total_blocks = blocks_y * blocks_x

    print(f"         总共 {blocks_y}x{blocks_x} = {total_blocks} 个块")

    processed_blocks = 0

    for by in range(blocks_y):
        for bx in range(blocks_x):
            # 计算块的边界（包含overlap）
            y_start = max(0, by * block_size - overlap)
            y_end = min(H, (by + 1) * block_size + overlap)
            x_start = max(0, bx * block_size - overlap)
            x_end = min(W, (bx + 1) * block_size + overlap)

            # 提取块（使用归一化后的数据）
            block_gx = Gx_norm[y_start:y_end, x_start:x_end]
            block_gy = Gy_norm[y_start:y_end, x_start:x_end]

            # 处理块
            denoised_gx = denoise_nl_means(block_gx, patch_size=patch_size,
                                          patch_distance=patch_distance, h=h,
                                          fast_mode=fast_mode)
            denoised_gy = denoise_nl_means(block_gy, patch_size=patch_size,
                                          patch_distance=patch_distance, h=h,
                                          fast_mode=fast_mode)

            # 计算实际写入区域（去除overlap）
            actual_y_start = by * block_size
            actual_y_end = min(H, (by + 1) * block_size)
            actual_x_start = bx * block_size
            actual_x_end = min(W, (bx + 1) * block_size)

            # 计算在块内的相对位置
            rel_y_start = actual_y_start - y_start
            rel_y_end = rel_y_start + (actual_y_end - actual_y_start)
            rel_x_start = actual_x_start - x_start
            rel_x_end = rel_x_start + (actual_x_end - actual_x_start)

            # 写入结果
            Gx[actual_y_start:actual_y_end, actual_x_start:actual_x_end] = \
                denoised_gx[rel_y_start:rel_y_end, rel_x_start:rel_x_end]
            Gy[actual_y_start:actual_y_end, actual_x_start:actual_x_end] = \
                denoised_gy[rel_y_start:rel_y_end, rel_x_start:rel_x_end]

            processed_blocks += 1

            # 进度更新
            if processed_blocks % max(1, total_blocks // 10) == 0:
                progress_pct = (processed_blocks / total_blocks) * 100
                elapsed = time.time() - start_time
                eta = elapsed * (total_blocks - processed_blocks) / processed_blocks if processed_blocks > 0 else 0
                print(f"         块进度: {progress_pct:.1f}% ({processed_blocks}/{total_blocks}), 耗时: {elapsed:.1f}s, 剩余: {eta:.1f}s")

                # 更新外部进度回调
                if progress_callback:
                    # 快速NLM在整个算法中占0.4-0.8的进度
                    nlm_progress = 0.4 + (processed_blocks / total_blocks) * 0.4
                    progress_callback(nlm_progress)

    total_time = time.time() - start_time
    print(f"      ✅ [fast_nlm] 分块处理完成，耗时: {total_time:.2f}s，处理了 {total_blocks} 个块")
    print(f"         平均每块耗时: {total_time/total_blocks:.3f}s")

    return Gx, Gy

# -------- Step 3：变分重建（在 [0,1] 量纲） --------
def variational_reconstruct_unit(R_unit, Gx, Gy,
                                 gamma=0.2, delta=0.8,
                                 iters=10, dt=0.15):
    I = R_unit.copy().astype(np.float32)
    for _ in range(iters):
        Ix, Iy = grad2d(I)
        grad_norm = np.sqrt(Ix*Ix + Iy*Iy) + 1e-12
        px = Ix / grad_norm; py = Iy / grad_norm
        div_p = div2d(px, py)
        lap_I = laplace(I)
        div_G = div2d(Gx, Gy)
        I = I - dt * (gamma * div_p + 2.0 * delta * (lap_I - div_G))
        I = np.clip(I, 0.0, 1.0)
    return I

# -------- 总封装：16-bit 进 → [0,1] 处理 → 16-bit 出 --------
def enhance_xray_poisson_nlm_strict(R16,
    # 归一化方式：percentile 更稳，window 用 DICOM WL/WW
    norm_mode="percentile", p_lo=0.5, p_hi=99.5, wl=None, ww=None,
    # Step1
    epsilon_8bit=2.3, mu=10.0, ksize_var=5,
    # Step2
    rho=1.5, search_radius=5, patch_radius=1, topk=None,
    count_target_mean=30.0, lam_quant=0.02,
    # Step3
    gamma=0.2, delta=0.8, iters=10, dt=0.15,
    # 输出
    out_dtype=np.uint16,
    # 进度回调
    progress_callback=None,
    # 快速模式
    use_fast_nlm=None  # None=自动判断, True=强制快速, False=使用原始实现
):
    import time
    start_time = time.time()
    H, W = R16.shape
    total_pixels = H * W

    print(f"   🔧 [enhance_xray_poisson_nlm_strict] 开始处理 {H}x{W} 图像 ({total_pixels:,} 像素)")

    # 大图像警告和参数自动调整
    if total_pixels > 2000000:  # 2M像素
        print(f"   ⚠️  检测到大图像 ({total_pixels/1000000:.1f}M像素)，自动调整参数以提高速度...")
        if search_radius > 1:
            search_radius = 1
            print(f"      - search_radius 调整为: {search_radius}")
        if topk is None or topk > 5:
            topk = 5
            print(f"      - topk 调整为: {topk}")
        if iters > 2:
            iters = 2
            print(f"      - iters 调整为: {iters}")
        if patch_radius > 1:
            patch_radius = 1
            print(f"      - patch_radius 调整为: {patch_radius}")

        # 超大图像(>5M像素)进一步优化
        if total_pixels > 5000000:
            print(f"   🚨 检测到超大图像 ({total_pixels/1000000:.1f}M像素)，使用极速模式...")
            search_radius = 1
            topk = 3
            iters = 1
            patch_radius = 1
            print(f"      - 极速参数: search_radius={search_radius}, topk={topk}, iters={iters}")
            print(f"      - 预计处理时间: {total_pixels*0.0005/60:.1f}分钟")

    # 16-bit → [0,1] 浮点（不丢精度）
    print(f"   📊 [normalize_to_unit] 开始归一化...")
    norm_start = time.time()
    R_unit, nctx = normalize_to_unit(R16, mode=norm_mode, p_lo=p_lo, p_hi=p_hi, wl=wl, ww=ww)
    print(f"   ✅ [normalize_to_unit] 完成，耗时: {time.time()-norm_start:.2f}s")

    # ε 从 8-bit 量纲换算到 [0,1]（注意：阈在 σ² 上 → 除以 255²）
    epsilon_unit = float(epsilon_8bit) / (255.0 * 255.0)

    if progress_callback:
        progress_callback(0.3)

    # Step1：梯度场增强
    print(f"   📊 [adaptive_gradient_enhance_unit] 开始梯度场增强...")
    step1_start = time.time()
    Gx_p, Gy_p = adaptive_gradient_enhance_unit(R_unit, epsilon_unit=epsilon_unit,
                                                mu=mu, ksize_var=ksize_var)
    print(f"   ✅ [adaptive_gradient_enhance_unit] 完成，耗时: {time.time()-step1_start:.2f}s")

    if progress_callback:
        progress_callback(0.4)

    # Step2：NLM处理（自动选择快速或原始实现）

    # 决定使用哪种NLM实现
    if use_fast_nlm is None:
        # 默认使用快速模式，因为原始实现性能太差
        use_fast_mode = True  # 默认总是使用快速模式
        print(f"      自动判断: 默认使用快速模式（原始实现性能较差）")
    else:
        use_fast_mode = use_fast_nlm
        print(f"      用户指定: 使用{'快速' if use_fast_mode else '原始'}模式")

    if use_fast_mode:
        print(f"   🚀 [fast_nlm] 使用快速NLM处理（基于skimage）...")
        print(f"      原因: {'自动检测大图像' if use_fast_nlm is None else '用户指定'}")
        step2_start = time.time()

        # 参数映射：search_radius -> patch_distance, patch_radius -> patch_size
        patch_size = max(3, patch_radius * 2 + 1)  # 最小3，1 -> 3, 2 -> 5
        patch_distance = max(5, search_radius * 2 + 3)  # 最小5，1 -> 5, 2 -> 7

        # 关键修复：大幅降低h值，避免过度平滑
        # 对于梯度场，h值需要非常小
        h = 0.001  # 从0.1降到0.001，避免过度去噪

        print(f"      🔧 参数映射: patch_size={patch_size}, patch_distance={patch_distance}, h={h}")

        Gx, Gy = fast_nlm_on_gradient(Gx_p, Gy_p,
                                      patch_size=patch_size,
                                      patch_distance=patch_distance,
                                      h=h, fast_mode=True,
                                      progress_callback=progress_callback)
    else:
        print(f"   📊 [poisson_nlm_on_gradient_exact] 使用原始泊松NLM处理...")
        print(f"      参数: search_radius={search_radius}, patch_radius={patch_radius}, topk={topk}")
        step2_start = time.time()
        Gx, Gy = poisson_nlm_on_gradient_exact(Gx_p, Gy_p,
                                               search_radius=search_radius,
                                               patch_radius=patch_radius,
                                               rho=rho,
                                               count_target_mean=count_target_mean,
                                               lam_quant=lam_quant,
                                               topk=topk,
                                               progress_callback=progress_callback)

    print(f"   ✅ [NLM处理] 完成，耗时: {time.time()-step2_start:.2f}s")

    if progress_callback:
        progress_callback(0.8)

    # Step3：变分重建（在 [0,1] 做）
    print(f"   📊 [variational_reconstruct_unit] 开始变分重建...")
    step3_start = time.time()
    I_unit = variational_reconstruct_unit(R_unit, Gx, Gy,
                                          gamma=gamma, delta=delta,
                                          iters=iters, dt=dt)
    print(f"   ✅ [variational_reconstruct_unit] 完成，耗时: {time.time()-step3_start:.2f}s")

    if progress_callback:
        progress_callback(0.9)

    # [0,1] → 16-bit（按归一化上下文反变换）
    print(f"   📊 [denormalize_from_unit] 开始反归一化...")
    denorm_start = time.time()
    I16 = denormalize_from_unit(I_unit, nctx, out_dtype=out_dtype)
    print(f"   ✅ [denormalize_from_unit] 完成，耗时: {time.time()-denorm_start:.2f}s")

    total_time = time.time() - start_time
    print(f"   🎉 [enhance_xray_poisson_nlm_strict] 总耗时: {total_time:.2f}s")

    return I16, (Gx_p, Gy_p), (Gx, Gy), nctx


