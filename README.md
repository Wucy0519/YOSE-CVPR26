# 🧊 YOSE: You Only Select Essential Tokens for Efficient DiT-based Video Object Removal

<a href='http://arxiv.org/abs/2604.27322'><img src='https://img.shields.io/badge/Paper-arxiv-b31b1b.svg'></a> &nbsp;
<a href="https://huggingface.co/datasets/wcy1234567/yose-dataset"><img alt="Huggingface TestDataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Benchmark-blue"></a> &nbsp;

This is PyTorch codes for our CVPR26 paper.

>**YOSE: You Only Select Essential Tokens for Efficient DiT-based Video Object Removal**<br>[Chenyang Wu<sup>1</sup>](), [ Lina Lei<sup>1</sup>](), [ Fan Li<sup>2</sup>](),[Chun-Le Guo<sup>1,3,&dagger;</sup>](), [Dehong Kong<sup>2</sup>](), [Xinran Qin<sup>2</sup>](), [Zhixin Wang<sup>2</sup>](), [Mingming Cheng<sup>1,3</sup>](), [Chongyi Li<sup>1,3</sup>]() <br>
> <sup>1</sup> VCIP, CS, Nankai University,  <sup>2</sup>Huawei Noah’s Ark Lab, <sup>3</sup>NKIARI, Shenzhen Futian<br>
>  <sup>&dagger;</sup>Corresponding author.

![teaser_img](assets/fig1.png)

![pipeline_img](assets/fig8.png)

### Introduction

Recent advances in Diffusion Transformer (DiT)-based video generation technologies have shown impressive results for video object removal.  However, these methods still suffer from substantial inference latency. For instance, although MiniMax Remover achieves state-of-the-art visual quality, it operates at only around 10 FPS, primarily due to dense computations over the entire spatiotemporal token space—even when only a small masked region actually requires processing. In this paper, we present **YOSE** — **Y**ou **O**nly **S**elect **E**ssential Tokens, an efficient fine-tuning framework. YOSE introduces two key components: Batch Variable-length Indexing (BVI) and Diffusion Process Simulator (DiffSim) Module.  BVI is a differentiable dynamic indexing operator that adaptively selects essential tokens based on mask information, enabling variable-length token processing across samples. DiffSim provides a diffusion process approximation mechanism for unmasked tokens, which simulates the influence of unmasked regions within DiT self-attention to maintain semantic consistency for masked tokens. With these designs, YOSE achieves mask-aware acceleration, where the inference time scales approximately linearly with the masked regions — in contrast to full-token diffusion methods whose computation remains constant regardless of the mask size. Extensive experiments demonstrate that YOSE achieves up to 2.5X speedup in 70% of cases while maintaining visual quality comparable to the baseline.

### More details are being organized

Coming Soon~

## Citation

```
@article{wu2026yose,
  title={YOSE: You Only Select Essential Tokens for Efficient DiT-based Video Object Removal},
  author={Wu, Chenyang and Lei, Lina and Li, Fan and Guo, Chun-Le and Kong, Dehong and Qin, Xinran and Wang, Zhixin and Cheng, Ming-Ming and Li, Chongyi},
  journal={arXiv preprint arXiv:2604.27322},
  year={2026}
}
```

## License

This project is licensed under the Pi-Lab License 1.0 - see the [LICENSE](LICENSE) file for details.
