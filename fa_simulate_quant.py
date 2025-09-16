# import torch
# a= torch.randn(1, 2, 3, device="npu", dtype=torch.int4)
# b = torch.randn(1, 2, 3, device="npu", dtype=torch.int4)
# c = a @ b


import torch
import math
import matplotlib.pyplot as plt
import numpy as np

Z = - 8 / 15
SCALE = 1/15
Count = None

def flash_attention_simulated(Q, K, V, block_M=128, block_N=128):
    """
    修改版实现：先计算Q块与完整K的注意力分数，再分块K进行online softmax
    
    参数:
    Q, K, V: BNSD格式输入 (B, H, S, D)
    block_M: Q块大小
    block_N: KV分块大小
    causal: 是否使用因果掩码
    
    返回:
    输出张量 (B, H, S, D)
    """
    B, H, S_q, D = Q.shape
    _, _, S_k, _ = K.shape
    scale = 1.0 / math.sqrt(D)
    # Q *= scale
    ori_dtype = Q.dtype
    # 初始化输出
    O = torch.zeros_like(Q)
    global Count, SCALE
    # 遍历Q块
    for i_start in range(0, S_q, block_M):
        i_end = min(i_start + block_M, S_q)
        q_block = Q[:, :, i_start:i_end]  # [B, H, M_i, D]
        M_i = i_end - i_start
        
        # === 关键修改：先计算Q块与完整K的注意力分数 ===
        # 计算Q块与完整K的注意力分数 (会创建M_i x S_k中间矩阵)
        s_full = torch.matmul(q_block, K.transpose(-2, -1)) * scale  # [B, H, M_i, S_k]
        # s_full = torch.matmul(q_block, K.transpose(-2, -1))   # [B, H, M_i, S_k]

        
        # === 分块处理K/V进行online softmax ===
        # 初始化当前Q块的归一化状态
        m_i = torch.full((B, H, M_i), -float('inf'), device=Q.device, dtype=torch.float32)
        l_i = torch.zeros((B, H, M_i), device=Q.device, dtype=torch.float32)
        o_i = torch.zeros((B, H, M_i, D), device=Q.device, dtype=torch.float32)
        
        # 遍历KV分块
        for j_start in range(0, S_k, block_N):
            j_end = min(j_start + block_N, S_k)
            N_j = j_end - j_start
            
            # 从完整分数中提取当前块
            # s_block = s_full[:, :, :, j_start:j_end]  # [B, H, M_i, N_j]
            s_block = s_full[:, :, :, j_start:j_end].to(torch.float32)  # [B, H, M_i, N_j]

            v_block = V[:, :, j_start:j_end]  # [B, H, N_j, D]
            
            # 更新最大值（数值稳定）
            s_max = torch.max(s_block, dim=-1).values  # [B, H, M_i]
            m_new = torch.maximum(m_i, s_max)
            
            # 计算指数项
            exp_fp16 = False
            if not exp_fp16:
                p = torch.exp(s_block - s_max.unsqueeze(-1))#.to(ori_dtype)
                assert p.dtype == torch.float32
            else:
                p = torch.exp(s_block.to(torch.float16) - s_max.unsqueeze(-1))

            p_int4 = p / SCALE
            p_int4 = p_int4.round().to(torch.int8)
            if Count is None:
                Count = torch.bincount(p_int4.view(-1), minlength=16)
            else:
                Count += torch.bincount(p_int4.view(-1), minlength=16)
            # 更新归一化项和输出
            l_new = torch.exp(m_i - m_new) * l_i + torch.exp(s_max - m_new) * p.sum(dim=-1)
            o_new = (torch.exp(m_i - m_new).unsqueeze(-1) * o_i + 
                    torch.exp(s_max - m_new).unsqueeze(-1) * torch.matmul(p.to(ori_dtype), v_block)).to(torch.float32)
            # 归一化
            # valid_mask = l_new > 0
            # o_new[valid_mask] = o_new[valid_mask] / l_new[valid_mask].unsqueeze(-1)
            
            # 传递状态到下一块
            m_i = m_new
            l_i = l_new
            o_i = o_new
        o_i = o_i / l_i.unsqueeze(-1)
        o_i = o_i.to(ori_dtype)
        # 存储结果
        O[:, :, i_start:i_end] = o_i
    
    return O


B, N, Q_S, D = 2, 5, 10000, 128
B, N, K_S, D = 2, 5, 10000, 128
q_shape = (B, N, Q_S, D)
k_shape = (B, N, K_S, D)

query = torch.randn(q_shape, dtype=torch.float16).cuda()
key = torch.randn(k_shape, dtype=torch.float16).cuda() 
v = torch.randn(k_shape, dtype=torch.float16).cuda()

B, N, S_q, D = query.shape
B, N, S_k, D = key.shape

res = flash_attention_simulated(query, key, v)
ref_res = torch.nn.functional.scaled_dot_product_attention(query, key, v,
                                        scale = D ** -0.5)
print(torch.allclose(res, ref_res,atol=.01, rtol=.01))
torch.testing.assert_close(res, ref_res, atol=.01, rtol=.01)

print(Count)


def plot_and_save_bincount(counts, output_path="histogram.png", title="Value Distribution", 
                          xlabel="Values", ylabel="Frequency"):
    """
    绘制并保存 torch.bincount 结果的直方图
    
    参数:
    counts -- torch.bincount 的结果
    output_path -- 保存图像的路径
    title, xlabel, ylabel -- 图表标题和标签
    """
    # 将 tensor 转换为 numpy 数组以便 matplotlib 处理
    counts_np = counts.cpu().numpy()
    
    # 创建 x 轴（值的范围）
    values = np.arange(len(counts_np))
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    
    # 绘制条形图（直方图） [[6]]
    plt.bar(values, counts_np, color='skyblue', edgecolor='black')
    
    # 添加标签和标题
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    # 如果值较多，旋转x轴标签防止重叠
    plt.xticks(values)
    
    # 添加网格线使图表更清晰
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 保存图像 [[4]]
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # 可选：显示图表（在非交互环境中可注释掉）
    # plt.show()
    
    print(f"直方图已保存至: {output_path}")


plot_and_save_bincount(Count, "histogram2.png" )
