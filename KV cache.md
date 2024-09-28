# KV cache

（内容源自知乎专栏[从0开始大模型学习——LLaMA2-KVcache详解 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/678523112)，本文仅供个人学习使用，非作者，在此声明转载链接）

## 大模型推理

一个典型的LLM的推理包含两个阶段：

Prefill Stage（预填充阶段）：输入一个prompt sequence生成KV cache。这个过程中存在大量GEMM（GEneral Matrix-Matrix multiply）操作，属于Compute-bound类型计算。

Decode Stage（解码生成阶段）：使用并更新KV cache，逐个生成并输出token，当前token的生成依赖于所有之前生成的token。GEMM变成GEMV操作，推理速度相对预填充阶段变快，这时属于Memory-bound类型计算。

## KV cache

Transformer模型具有自回归推理的特点，即每次推理只会预测输出一个token，当前轮输出token与历史输入tokens拼接，作为下一轮的输入token，反复执行。在该过程中，前后两轮的输入只相差一个token，存在重复计算。KV cache技术实现了将可复用的K和V向量保存下来，避免重复计算。

大模型的Decode Stage阶段，涉及到attention的计算。attention的计算公式如下：
$$
attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$
假设此时已经输入了n个tokens，每个token经过embedding后转换为一个对应的向量（1 x dim）。此时的Q（n x dim）是由n个embedding向量组成的矩阵和训练好的Qw矩阵相乘得到的，K（n x dim）也一样。将Q和K的转置相乘以得到包含各个token注意力关系的矩阵（比如某token1和token3关系紧密，于是矩阵的第三行第一列的值较大，由于mask机制的存在，计算时只会计算矩阵的下半区域，也即每个token只会与历史的tokens进行注意力的计算）。然后对矩阵每个元素除以d_k（由于LLM并行的机制，实际训练时会将dim均分为多个相等的dim_k，然后在k个模块上同时进行训练，最后将结果矩阵进行简单拼接），这一步是为了平衡数字分布，防止softmax后矩阵中某一行出现一个数据是99%，其他数据是0.xx%的情况。将矩阵进行softmax后再乘以V（V也是由embedding x 训练后得到的参数Vw计算得到的） ，我们计算得出了attention矩阵。下图是attention计算过程的示意图：

![attention](D:\LLM_storage\images\attention.jpg)

在下图中，第四个attention的计算只依赖于QK^T的最后一行，具体地，只依赖于第四个token的q向量与四个tokens的K向量和V向量。所以在大模型的推理阶段，不需要缓存q向量。这解释了KV cache名字的含义，只需缓存k向量和v向量。

![attention_analysis](D:\LLM_storage\images\attention_analysis.jpg)

同时，在计算第二个attention时，token2的k是由token2的embedding与训练好的kw矩阵相乘得到的，token2的v是由token2的embedding与训练好的kv矩阵相乘得到的。于是K、V矩阵可以由之前KV cache缓存的token1的k、v向量加上token2的k、v向量拼接而成。下面是应用了KV cache后token2计算attention的过程。

![the impact of KV cache](D:\LLM_storage\images\the impact of KV cache.jpg)

在得到了attention2后进行解码可以得到token3，以此类推。

## 涉及到的术语的解释

1、prompt：通常指一种输入方式，即通过特定的指令或问题来引导模型进行特定的输出或任务。比如，用户向模型提供的输入文本或指令，可以是一个问题、一段描述、一个任务说明，甚至是一部分对话历史记录等等。

2、GEMM：通用矩阵x矩阵乘法。

3、Compute-bound：计算密集型，主要指某个操作或任务在执行过程中，其大部分时间被用于执行CPU密集型的操作，即CPU的运算负担较重，比如涉及大量计算、逻辑判断等。在执行这类任务时，系统主要特点有：CPU负载高、I/O响应快、计算量大。

4、token：在NLP任务中，一个token通常是一个词或者是文本中的最小单元。

5、GEMV：通用矩阵x向量乘法。

6、Memory-bound：内存受限。主要指某个操作或任务在执行过程中，其性能或速度主要受限于内存（包括缓存和主存）的访问速度和容量。这类操作通常涉及大量的数据读写操作，而CPU的计算能力相对充裕，但受限于内存带宽和数据访问延迟，导致整体性能无法充分发挥。在深度学习领域，大模型的推理过程往往涉及大量的参数和数据加载，这些参数和数据通常存储在外部存储器（如GPU显存或CPU内存）中。当模型规模增大时，单次推理所需加载的数据量也会增加，从而可能导致内存带宽成为瓶颈，使得推理速度受限。

7、softmax：机器学习常用函数之一，可以将一个含任意实数的K维向量压缩到另一个K维向量中，使得每一个元素的范围都在(0, 1)之间，且全部元素的和为1。对矩阵进行softmax，意味着对矩阵的每一行进行softmax。简单理解，就是将每个元素所占每一行数值之和的百分比计算出来得到原矩阵的softmax矩阵。

8、casual：Causal Decoder是一种模型架构，特别适用于生成文本的任务，如写故事、文章续写或问答系统等。它确保了生成的文本在顺序上符合因果关系，即每个词的生成仅依赖于它之前的所有词。

