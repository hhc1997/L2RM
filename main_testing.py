import os
from evaluation import evalrank, evalrank_SGRAF, evaluation



if __name__ == "__main__":
    # load arguments
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    data_name = 'coco_precomp'
    data_path = './data'
    vocab_path = './vocab'
    model_path_SAF = './model_best.pth.tar'
    model_path_SGR = './model_best.pth.tar'

    #evaluation(model_path=[model_path_SAF,model_path_SGR], data_path=None, split='testall', fold5=True)
    evaluation(model_path=[model_path_SAF, model_path_SGR], data_path=None, split='test', fold5=False)