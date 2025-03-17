# MMRL: Multi-Modal Representation Learning for Vision-Language Models  

This repository provides the official PyTorch implementation for the CVPR 2025 paper:  
**MMRL: Multi-Modal Representation Learning for Vision-Language Models**  

📄 [arXiv Paper Link](https://arxiv.org/abs/2503.08497)  

## 🔧 Installation  

MMRL is built upon [CoOp](https://github.com/KaiyangZhou/CoOp) and [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning).  

To set up the runtime environment and install the required datasets, please follow the instructions in the [CoOp repository](https://github.com/KaiyangZhou/CoOp). We sincerely appreciate their contributions!  

## 🚀 Running the Code  

We provide various scripts for different experimental settings. The main scripts are:

- `base_to_novel.sh` (Base-to-Novel Generalization)
- `cross_datasets.sh` (Cross-Dataset Evaluation and Domain Generalization)
- `few_shot.sh` (Few-Shot Learning)
- Detailed bash scripts in `scripts/mmrl`

To run the experiments, navigate to the MMRL root directory and execute the corresponding script. Make sure to replace `DATA` with the path to your dataset in `scripts/mmrl`.  
### **Base-to-Novel Generalization**  

Run the following command:  

```bash
bash base_to_novel.sh
```

You can modify configurations in:  
- `trainer/config.py`  
- `configs/trainers/MMRL/vit_b16.yaml`  
- `configs/trainers/MMRL/vit_b16_imagenet.yaml`  

### **Cross-Dataset Evaluation and Domain Generalization**  

Run the following command:  

```bash
bash cross_datasets.sh
```

You can adjust configurations in:  
- `configs/trainers/MMRL/vit_b16_cross_datasets.yaml`  
- `scripts/mmrl/cross_datasets_train.sh`  
- `scripts/mmrl/cross_datasets_test.sh`  

**Note:** Ensure that the `REP_DIM` value remains consistent between training on ImageNet and testing on other datasets.  

### **Few-Shot Learning**  

Run the following command:  

```bash
bash few_shot.sh
```

Configurations can be adjusted in:  
- `trainer/config.py`
- `configs/trainers/MMRL/vit_b16_few_shot.yaml`
- `configs/trainers/MMRL/vit_b16_imagenet.yaml`

## 📌 Citation  

If you find this repository useful for your research, please consider citing:  

```bibtex
@misc{guo2025mmrlmultimodalrepresentationlearning,
      title={MMRL: Multi-Modal Representation Learning for Vision-Language Models}, 
      author={Yuncheng Guo and Xiaodong Gu},
      year={2025},
      eprint={2503.08497},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.08497}, 
}
```