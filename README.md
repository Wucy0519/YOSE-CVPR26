# :fire: YOSE: You Only Select Essential Tokens for Efficient DiT-based Video Object Removal

This is the official PyTorch codes for our CVPR26 paper.

>**YOSE: You Only Select Essential Tokens for Efficient DiT-based Video Object Removal**<br>[Chenyang Wu<sup>1</sup>](), [ Lina Lei<sup>1</sup>](), [ Fan Li<sup>2</sup>](),[Chun-Le Guo<sup>1,3,&dagger;</sup>](), [Dehong Kong<sup>2</sup>](), [Xinran Qin<sup>2</sup>](), [Zhixin Wang<sup>2</sup>](), [Mingming Cheng<sup>1,3</sup>](), [Chongyi Li<sup>1,3</sup>]() <br>
> <sup>1</sup> VCIP, CS, Nankai University,  <sup>2</sup>Huawei Noah’s Ark Lab, <sup>3</sup>NKIARI, Shenzhen Futian<br>
>  <sup>&dagger;</sup>Corresponding author.

![teaser_img](assets/fig1.png)

![pipeline_img](assets/fig8.png)

### Introduction

Recent advances in Diffusion Transformer (DiT)-based video generation technologies have shown impressive results for video object removal.  However, these methods still suffer from substantial inference latency. For instance, although MiniMax Remover achieves state-of-the-art visual quality, it operates at only around 10 FPS, primarily due to dense computations over the entire spatiotemporal token space—even when only a small masked region actually requires processing. In this paper, we present **YOSE** — **Y**ou **O**nly **S**elect **E**ssential Tokens, an efficient fine-tuning framework. YOSE introduces two key components: Batch Variable-length Indexing (BVI) and Diffusion Process Simulator (DiffSim) Module.  BVI is a differentiable dynamic indexing operator that adaptively selects essential tokens based on mask information, enabling variable-length token processing across samples. DiffSim provides a diffusion process approximation mechanism for unmasked tokens, which simulates the influence of unmasked regions within DiT self-attention to maintain semantic consistency for masked tokens. With these designs, YOSE achieves mask-aware acceleration, where the inference time scales approximately linearly with the masked regions — in contrast to full-token diffusion methods whose computation remains constant regardless of the mask size. Extensive experiments demonstrate that YOSE achieves up to 2.5X speedup in 70% of cases while maintaining visual quality comparable to the baseline.

### Core Algorithm 

#### Batch Variable-length Indexing (A Part of Core Codes)

````python
def get_index_grad_batch(index):
    # index: [b, num]
    b, num = index.shape[0], index.shape[1]
    delta_tmp = 1/(2*num)

    tmp = torch.linspace(delta_tmp-1.0, 1.0-delta_tmp, num).to(index.device).float()

    final_ = []
    final_len = []
    backward_ = []

    for i in range(b):
        short_index = tmp[index[i]>0.5] # short_index: [ short_num ]
        short_num = short_index.shape[0]

        final_.append(short_index)
        final_len.append(short_num)

    max_len = max(final_len)
    delta_max = 1/(2*max_len)
    for i in range(b):
        tmp_short = torch.linspace(delta_max-1.0, ((2-2*delta_max)/(max_len-1))*(final_len[i]-1)-1+delta_max, final_len[i]).to(index.device).float()
        this_tmp = tmp.clone()
        this_tmp[index[i]>0.5] = tmp_short
        backward_.append(this_tmp)

        if max_len != final_len[i]:
            add_tensor = torch.ones([max_len-final_len[i]]).to(final_[i].device)
            final_[i] = torch.cat([final_[i], add_tensor], dim=0)

    forward_index = torch.stack(final_, dim=0)          # [b, short_num]
    backward_index = torch.stack(backward_, dim=0)      # [b, orgin_num]
    return forward_index, backward_index 

def index1d_batch(x, index):
    # x    : [b, num, c]
    # index: [b, other_num]
    b, other_num = index.shape
    xx = x.permute(0, 2, 1).unsqueeze(3) # [b, c, num, 1]
    index_ = index.reshape([b, other_num, 1, 1]) # [b, other_num, 1, 1]
    index_ = torch.cat([index_*0, index_], dim=-1).to(x.device)
    out = F.grid_sample(input=xx.float(), grid=index_.float(), mode='nearest', padding_mode='reflection')
    return out.squeeze(3).permute(0, 2, 1).type_as(x)

if __name__ == "__main__":
    inpu = torch.randn([4, 7800, 1536]).cuda()
    masks = torch.randn([4, 7800]).cuda()

    forward_index, backward_index = get_index_grad_batch(masks*0+1)
    
    short_ = index1d_batch(inpu, forward_index)
    long_ = index1d_batch(short_, backward_index)
````

### More details are being organized

Coming Soon~

## License

This project is licensed under the Pi-Lab License 1.0 - see the [LICENSE](LICENSE) file for details.
