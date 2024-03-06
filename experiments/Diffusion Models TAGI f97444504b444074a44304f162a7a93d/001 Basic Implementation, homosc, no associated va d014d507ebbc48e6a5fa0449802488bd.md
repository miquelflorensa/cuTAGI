# 001 Basic Implementation, homosc, no associated variance

# Results

## Swiss Roll

- Batch: 2000
- 40 diffusion steps
- Cap factor: 1
- Noise gain: 0.1
- Sigma_v: 2
- Sigma_v_min: 0.1
- Decay Factor: 0.99
- Training time CUDA 100 epochs: 17s
- 100.000 points for training and 10.000 for sampling.

![diffusion_swiss_roll-1.png](001%20Basic%20Implementation,%20homosc,%20no%20associated%20va%20d014d507ebbc48e6a5fa0449802488bd/diffusion_swiss_roll-1.png)

SwissRoll pytorch Implementation from **[albarji](https://github.com/albarji/toy-diffusion/commits?author=albarji)**

- Batch: 2048
- Training time CUDA 100 epochs: 30s

![5f845a9c-c954-47fc-b477-ff00890a91fb.png](001%20Basic%20Implementation,%20homosc,%20no%20associated%20va%20d014d507ebbc48e6a5fa0449802488bd/5f845a9c-c954-47fc-b477-ff00890a91fb.png)

Predicted variance in last sampling timestep with TAGI DM:

![diffusion_swiss_roll_variance-1.png](001%20Basic%20Implementation,%20homosc,%20no%20associated%20va%20d014d507ebbc48e6a5fa0449802488bd/diffusion_swiss_roll_variance-1.png)

Error variance during training: 

TAGI DM

Toy problem pyTorch

![error_variance.png](001%20Basic%20Implementation,%20homosc,%20no%20associated%20va%20d014d507ebbc48e6a5fa0449802488bd/error_variance.png)

![62699744-c253-40a7-aa9f-544b0d3dfebe.png](001%20Basic%20Implementation,%20homosc,%20no%20associated%20va%20d014d507ebbc48e6a5fa0449802488bd/62699744-c253-40a7-aa9f-544b0d3dfebe.png)

3 first epochs descend from 0.01 to almost 0.

### Training times:

- Batch size 2048.

|  | CPU$^{(1)}$ | CUDA$^{(2)}$  | Colab T4 | MPS$^{(4)}$  |
| --- | --- | --- | --- | --- |
| TAGI | 434.4 sec. | 17 sec. | - | - |
| TORCH | 12.6 sec. ??$^{(3)}$ | 30 sec. | 14 sec. | 37.5 sec. |

$^{(1)}$ *CPU: Macbook M2 8 CPU cores, 24 GB Ram.*

$^{(2)}$ *CUDA: 2 x NVIDIA Quadro RTX 5000*

$^{(3)}$ *See performance M2:* [https://www.lightly.ai/post/apple-m1-and-m2-performance-for-training-ssl-models](https://www.lightly.ai/post/apple-m1-and-m2-performance-for-training-ssl-models) 

$^{(4)}$ *MPS: Macbook M2 10 GPU cores, 24 GB Ram.*

- Batch size 2048*10.

|  | CPU$^{(1)}$ | CUDA$^{(2)}$  | Colab T4 | MPS$^{(4)}$  |
| --- | --- | --- | --- | --- |
| TAGI (BS 2048*3) | - | 16 sec. | - | - |
| TORCH | 8 sec. | 2 sec.  | 4 sec. | 6.2 sec. |

* *For some reason the Server seems to be using a single GPU* 

## Moon Dataset

- Batch: 2000
- Cap factor: 1
- Noise gain: 0.1
- Sigma_v: 2
- Sigma_v_min: 0.1
- Decay Factor: 0.99
- Training time CUDA 100 epochs: 17s

![diffusion_swiss_roll.png](001%20Basic%20Implementation,%20homosc,%20no%20associated%20va%20d014d507ebbc48e6a5fa0449802488bd/diffusion_swiss_roll.png)

![diffusion_swiss_roll_variance.png](001%20Basic%20Implementation,%20homosc,%20no%20associated%20va%20d014d507ebbc48e6a5fa0449802488bd/diffusion_swiss_roll_variance.png)