# Generalizing to Unseen Speakers: Multimodal Emotion Recognition in Conversations with Speaker Generalization

Link for the paper:
[Generalizing to Unseen Speakers: Multimodal Emotion Recognition in Conversations with Speaker Generalization](https://ieeexplore.ieee.org/abstract/document/10981602), TAFFC 2025.

### Requirements

- Python 3.8.5
- torch 1.7.1
- CUDA 11.3
- torch-geometric 1.7.2

### Dataset

The raw data can be found at [IEMOCAP](https://sail.usc.edu/iemocap/ "IEMOCAP") and [MELD](https://github.com/SenticNet/MELD "MELD").

In our paper, we use pre-extracted features. The multimodal features (including RoBERTa-based and GloVe-based textual features) are available at [here](https://www.dropbox.com/sh/4b21lympehwdg4l/AADXMURD5uCECN_pvvJpCAy9a?dl=0 "here").

### Checkpoints

The implementation results may vary with training machines and random seeds. We suggest that one can try different random seeds for better results.


### Training examples

To train on IEMOCAP:

bash train_IEMOCAP.sh

To train on MELD:

bash train_MELD.sh

## Reference

If you found this code useful, please cite the following paper:
```
@ARTICLE{10981602,
  author={Tu, Geng and Jing, Ran and Liang, Bin and Yu, Yue and Yang, Min and Qin, Bing and Xu, Ruifeng},
  journal={IEEE Transactions on Affective Computing}, 
  title={Generalizing to Unseen Speakers: Multimodal Emotion Recognition in Conversations With Speaker Generalization}, 
  year={2025},
  volume={},
  number={},
  pages={1-12},
  keywords={Uncertainty;Prototypes;Emotion recognition;Context modeling;Electronic mail;Visualization;Training;Oral communication;Graph neural networks;Data models;Contrastive learning;multimodal conversational emotion recognition;prototype graph;speaker generalization},
  doi={10.1109/TAFFC.2025.3566059}
}
```
