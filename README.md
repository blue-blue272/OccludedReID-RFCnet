## [Feature Completion for Occluded Person Re-Identification](https://arxiv.org/pdf/2106.12733)

### Training and test

  ```Shell
  1. we first generate the part masks with the code https://github.com/Engineering-Course/LIP_JPPNet/.
  2. python train.py --root "your path to the dataset" --fore_dir "your path to extracted foremaps"
  3. python train.py --root "your path to the dataset" --fore_dir "your path to extracted foremaps" --resume "path to model.pth" --evaluate
  ```
  
  
### Citation
If you use our code in your research, please use the following BibTeX entry.

   @inproceedings{hou2021RFCnet,
      title={Feature Completion for Occluded Person Re-Identification},
      author={Ruibing Hou and Bingpeng Ma and Hong Chang and Xinqian Gu and Shiguang Shan and Xilin Chen},
      journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
      year={2021},
      publisher={IEEE}
}

```
