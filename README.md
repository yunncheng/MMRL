# MMRL: Multi-Modal Representation Learning for Vision-Language Models (CVPR2025) & MMRL++: Parameter-Efficient and Interaction-Aware Representation Learning for Vision-Language Models (arXiv)

This repository provides the official PyTorch implementation for our CVPR 2025 paper:  
**MMRL: Multi-Modal Representation Learning for Vision-Language Models**  
and our arXiv extension:  
**MMRL++: Parameter-Efficient and Interaction-Aware Representation Learning for Vision-Language Models**

📄 [MMRL Paper Link](https://arxiv.org/abs/2503.08497)  
📄 [MMRL++ Paper Link](https://arxiv.org/abs/2505.10088)




## 📢 News

- 🗓️ 2025/05/21: MMRL++ code is released!  
- 🗓️ 2025/05/15: MMRL++ arXiv version is available.  
- 🗓️ 2025/03/11: MMRL arXiv version is available.  
- 🗓️ 2025/03/04: MMRL code is released!  
- 🗓️ 2025/02/27: MMRL is accepted by CVPR 2025 🎉  



## 🔧 Installation  

MMRL and MMRL++ build upon [CoOp](https://github.com/KaiyangZhou/CoOp) and [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning). Please refer to the [CoOp](https://github.com/KaiyangZhou/CoOp) repository for dataset setup instructions. We sincerely appreciate their contributions!

To set up the runtime environment, you can follow the guidelines provided in the [CoOp](https://github.com/KaiyangZhou/CoOp) repository or use the step-by-step instructions below (recommended) to create and configure your environment.

* Setup conda environment (recommended).
```bash
# Create a conda environment
conda create -y -n mmrl python=3.10

# Activate the environment
conda activate mmrl

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

* Install [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch) library.
```bash
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation

# Clone this repo
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
cd ..
```

* Clone MMRL code repository
```bash
git clone https://github.com/yunncheng/MMRL.git
cd MMRL/
```



## 🚀 Running the Code  

We provide various scripts for different experimental settings. The main scripts are:

- `base_to_novel.sh` (Base-to-Novel Generalization)
- `cross_datasets.sh` (Cross-Dataset Evaluation and Domain Generalization)
- `few_shot.sh` (Few-Shot Learning)
- Detailed bash scripts in `scripts/mmrl` and `scripts/mmrlpp`

To run the experiments, navigate to the MMRL root directory and execute the corresponding script. Make sure to replace `DATA` with the path to your dataset in `scripts/mmrl` and `scripts/mmrlpp`.  
### **Base-to-Novel Generalization**  

Run the following command:  

```bash
bash base_to_novel.sh
```

You can modify configurations in:  
- `trainer/config.py`  
- `configs/trainers/MMRL/vit_b16.yaml`  
- `configs/trainers/MMRL/vit_b16_imagenet.yaml`  
- `configs/trainers/MMRLpp/vit_b16.yaml`  
- `configs/trainers/MMRLpp/vit_b16_imagenet.yaml`  

### **Cross-Dataset Evaluation and Domain Generalization**  

Run the following command:  

```bash
bash cross_datasets.sh
```

You can adjust configurations in:  
- `trainer/config.py`  
- `configs/trainers/MMRL/vit_b16_cross_datasets.yaml`  
- `configs/trainers/MMRLpp/vit_b16_cross_datasets.yaml`  
- `scripts/mmrl/cross_datasets_train.sh`  
- `scripts/mmrl/cross_datasets_test.sh`   

**Note:** Ensure that the `REP_DIM` value remains consistent between training on ImageNet and testing on other datasets when runing MMRL.  

### **Few-Shot Learning**  

Run the following command:  

```bash
bash few_shot.sh
```

Configurations can be adjusted in:  
- `trainer/config.py`
- `configs/trainers/MMRL/vit_b16_few_shot.yaml`
- `configs/trainers/MMRL/vit_b16_imagenet.yaml`
- `configs/trainers/MMRLpp/vit_b16_few_shot.yaml`
- `configs/trainers/MMRLpp/vit_b16_imagenet.yaml`



## ✨ MMRL++

MMRL++ is an extension of MMRL that introduces:
- **Shared-Residual Representation Aligner (SRRA):** A parameter-efficient design for gradient and information sharing.
- **Progressive Representation Composition (PRC):** Enhances intra-modal interaction via inter-layer instance-specific semantic flow.

It achieves stronger generalization with fewer trainable parameters while maintaining or improving performance across multiple benchmarks.

📄 Read the MMRL++ paper here: [https://arxiv.org/abs/2505.10088](https://arxiv.org/abs/2505.10088)



## 🧩 Model Zoo

You can find the trained MMRL and MMRL++ model weights and corresponding log files at [Model / Logs](https://drive.google.com/drive/folders/1z_iKB8bNCzpZHI_jf_cWzrAn0J8d_5-Y?usp=sharing).

**Please Note:** We have fixed some naming bugs in the code while uploading the weights. Therefore, if you wish to use our trained weights, please ensure you are using the latest open-source code.



## 📌 Citation  

If you find this repository useful for your research, please consider citing:  

```bibtex
@inproceedings{guo2025mmrl,
      title={Mmrl: Multi-modal representation learning for vision-language models},
      author={Guo, Yuncheng and Gu, Xiaodong},
      booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
      pages={25015--25025},
      year={2025}
}

@misc{guo2025mmrlparameterefficientinteractionawarerepresentation,
      title={MMRL++: Parameter-Efficient and Interaction-Aware Representation Learning for Vision-Language Models}, 
      author={Yuncheng Guo and Xiaodong Gu},
      year={2025},
      eprint={2505.10088},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.10088}, 
}
```