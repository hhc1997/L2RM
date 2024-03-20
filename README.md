PyTorch implementation for "Learning to Rematch Mismatched Pairs for Robust Cross-Modal Retrieval (paper ID: 5638)"
# Learning to Rematch Mismatched Pairs for Robust Cross-Modal Retrieval

## Requirements
- Python 3.8
- torch 1.12
- numpy
- scikit-learn
- pomegranate [Install](https://github.com/jmschrei/pomegranate/pull/901)
- Punkt Sentence Tokenizer:
  
```
import nltk
nltk.download()
> d punkt
```
## Datasets
We follow [NCR](https://github.com/XLearning-SCU/2021-NeurIPS-NCR) to obtain image features and vocabularies.

[Download Dataset](https://ncr-paper.cdn.bcebos.com/data/NCR-data.tar)

## Noise (Mismatching) Index
We use the same noise index settings as [DECL](https://github.com/QinYang79/DECL) and [RCL](https://github.com/penghu-cs/RCL), which could be found in ```noise_index```. The mismatching ratio (noise ratio) is set as 0.2, 0.4, 0.6, and 0.8.


## Training and Evaluation
### Training new models
Modify some necessary parameters and run it.

For Flickr30K:
```
sh train_f30k.sh
```

For MS-COCO:
```
sh train_coco.sh
```

For CC152K:
```
sh train_cc152k.sh
```

### Evaluation
Modify some necessary parameters and run it.
```
python main_testing.py
```

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)



## Acknowledgements
The code is based on [SCAN](https://github.com/kuanghuei/SCAN), [SGRAF](https://github.com/Paranioar/SGRAF), [NCR](https://github.com/XLearning-SCU/2021-NeurIPS-NCR), [DECL](https://github.com/QinYang79/DECL), and [KPG-RL](https://github.com/XJTU-XGU/KPG-RL) licensed under Apache 2.0. 

