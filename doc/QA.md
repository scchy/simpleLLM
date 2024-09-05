# 一、About Transformer 

## Q: FFN的作用

1. 提供更加纯粹的非线性变化
   - $FFN = Linear(RelU(Linear(Y^{Atten})))$
   - Attention本质上是Value的线性组合 $Y^{Atten}=\sum atten *  X W^T_V$
2. FFN其实是Key-Value记忆
   1. $Y=\sigma (X W_K^T)W_V^T$
      1. [paper-2015: End-To-End Memory Networks](https://arxiv.org/pdf/1503.08895)
      2. [paper-2020: <EMNLP 2021> Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913)
   2. 第一层key-memory 是对词的一些partition的记忆
   3. 第二层Value-memory 是对句子整体的partition的记忆，比如语义，分类等
   4. Att 是对短期的信息进行提取，而FFN是对整个样本进行信息提取和记忆
3. FFN的参数占比 66%以上
```python
# FFN
ffn_ = gpt2.transformer.h[0].mlp
ffn_MB = sum([t.numel() * 4 for t in ffn_.parameters()])/1024**2

att_ = gpt2.transformer.h[0].attn
att_MB = sum([t.numel() * 4 for t in att_.parameters()])/1024**2
ffn_MB / (att_MB + ffn_MB) # 66.6%
```

## Q: pre-Norm 和 post-Norm的差异

1. 公式上的差异
   1. pre-Norm
      1. Y = Atten(LN(X)) + X
      2. O = FFN(LN(Y)) + Y
   2. post-Norm
      1. Y = LN(Atten(X) + X)
      2. O = LN(FFN(Y) + Y)
2. pre norm训练更加的稳定
   1. 实验表明post norm 不进行warm up训练难以收敛，亦或收敛不好。
      1. [paper-2020 <EMNLP 2021> Understanding the difficulty of training transformers](https://arxiv.org/pdf/2004.08249)
   2. 原因可以简单推导：对于两个均值为0，方差为1且相互独立的分布 `[A, B]`
      1. post norm: 
         1. $Y = LN(Atten(X) + X) = \frac{(Atten(X) + X)-0}{\sqrt{2}}$
         2. $O=\frac{FFN(Y) + Y}{\sqrt{2}} = \frac{FFN(Y) + \frac{Atten(X) + X}{\sqrt{2}}}{\sqrt{2}}=Z+\frac{X}{2}$
         3. 当层数增多$O^2=\frac{FFN(Y') + \frac{Atten(O) + O}{\sqrt{2}}}{\sqrt{2}}=Z'+\frac{X}{2^2}$; $O^{12}=Z'+\frac{X}{2^{12}}$ 
         4. 即resNet几乎不起作用了
      2. pre Norm:
         1. $Y = Atten(\frac{X}{\sqrt{2}}) + X$
         2. $O = FFN(LN(Y)) + Atten(\frac{X}{\sqrt{2}}) + X=Z + X$
         3. 不论多少层resNet依然起作用
      3. 所以现在基于transformer架构的大模型全都是采用preNorm


## Q: Attention 中$\sqrt{K_{dim}}$的作用是什么

1. 相当于temperature-softmax=$\frac{e^{X/T}}{\sum e^{X/T}}$
   1. $T=\sqrt{K_{dim}} > 0$ softmax会更加平滑有利于模型学习
   2. 避免softmax退化成argmax 
2. 简单验证: 假设每个dim 均值为0，方差为1
   1. $a=\sum_{i=0}^{d}{q_i}{k_i}; var(a)=\sum_{i=0}^{d}{var(q_i)}{var(k_i)}=\sum_{i=0}^{d} 1 = d$
   2. 即如果不除以$\sqrt{K_{dim}}$ a的标准差$std(a)=\sqrt{var(a)}=\sqrt{d}$ 会随dim增加而增加, 最终退化成argmax


## Q: 为什么Transformer用Layer Norm 而不是 batch norm

1. batch norm 在LLM任务中不适用
   1. inference的时候样本量和train的时候不一致，影响inference
   2. 分布式训练采用batch norm的话需要额外的交互开销
   3. NLP任务中多存在padding影响batchNorm
2. Layer Norm 有利于模型的训练
   1. [paper-2023 On the expressivity Role of LayerNorm in Transformers' Attention](https://arxiv.org/pdf/2305.02582)
   2. 映射到 query和key正交的超平面，这样方便所有的key被同等访问
   3. scaling 每个key都能被选中，都有机会获得最高分

## Q: position embedding 的作用



