import numpy as np 
import torch 

"""

"""

class DiversityMeasure():

    def __init__(self, mode='standard') -> None:
        """
        preds: list of predictions, form of probability
                shape of each pred: [batch_size, num_classes]
        targets: [batch_size,]
        """
        self.mode = mode 
    
    def one_metric(self, preds, targets, metric): 
        if metric == 'disagreement': 
            return disagreement(preds)
        elif metric == 'norm_disagreement': 
            return norm_disagreement(preds, targets)
        elif metric == 'double_fault_measure': 
            return double_fault_measure(preds, targets)
        elif metric == 'pearson_correlation': 
            return pearson_correlation(preds)
        elif metric == 'cosine_similarity': 
            return cosine_similarity(preds)
    
    def standard(self, preds, targets): 
        metrics = ['disagreement', 
                'norm_disagreement',
                'double_fault_measure', 
                'pearson_correlation',
                'cosine_similarity']
        output = dict()
        for m in metrics: 
            output[m] = self.one_metric(preds, targets, m)
        return output

def pair_disagreement(pred_i, pred_j): 
    """
        Disagreement between a pair of preds 
    """
    output = torch.eq(torch.argmax(pred_i, dim=-1), torch.argmax(pred_j, dim=-1))
    return torch.sum(output.float())/output.shape[0]

def disagreement(preds): 
    """
    Ref: http://ceur-ws.org/Vol-2916/paper_8.pdf 
    Disagree of pair classifiers: D_{i,j} = N^{\tilde{y_i} \neq \tilde{y_j}} / N 
    \tilde{y_i}: predicted label of classifier i 
    \tilde{y_j}: predicted label of classifier j 
    Args: 
        - preds: list of prediction [pred, ... ,pred], each pred has shape [batch_size, num_classes]
    """
    assert(len(preds) >= 2)

    output = 0 
    num_pairs = 0 
    for i, pred_i in enumerate(preds): 
        for j, pred_j in enumerate(preds): 
            if i == j: 
                continue
            num_pairs += 1 
            output += pair_disagreement(pred_i, pred_j) 
    return output / num_pairs

def ensemble_acc(pred1, pred2, targets): 
    """
    Ensemble Accuracy of two predictions. Ensemble = (pred1 + pred2)/2
    Args: 
        pred1, pred2 
        targets 
    """
    combine_pred = (pred1 + pred2)/2
    output = torch.eq(targets, torch.argmax(combine_pred, dim=-1))

    return torch.sum(output.float()) / output.shape[0]

def norm_disagreement(preds, targets): 
    """
    Ref: http://ceur-ws.org/Vol-2916/paper_8.pdf 
    Normalized Disagreement 
    ND_{i,j} = D_{i,j} / (1 - accuracy_{i,j})
    """
    assert(len(preds) >= 2) 
    output = 0 
    num_pairs = 0 
    for i, pred_i in enumerate(preds): 
        for j, pred_j in enumerate(preds): 
            if i == j: 
                continue
            num_pairs += 1 
            acc = ensemble_acc(pred_i, pred_j, targets)
            disagree = pair_disagreement(pred_i, pred_j)
            output += disagree / (1 - acc)

    return output / num_pairs 

def pair_double_fault(pred_i, pred_j, targets): 
    output1 = torch.eq(targets, torch.argmax(pred_i, dim=-1)).float() # [0,1]
    output2 = torch.eq(targets, torch.argmax(pred_j, dim=-1)).float() # [0,1]
    output = (output1 + output2) < 2 
    return torch.mean(output.float())


def double_fault_measure(preds, targets): 
    """"
    DF_{i.j} = N^{00}_{i,j}/N 
    where N^{00}_{i,j} is number of samples that both members make wrong predictions 
    """
    assert(len(preds) >= 2)
    num_pairs = 0 
    for i, pred_i in enumerate(preds): 
        for j, pred_j in enumerate(preds): 
            if i == j: 
                continue
            num_pairs += 1 
            output = pair_double_fault(pred_i, pred_j, targets)
    return output / num_pairs

def pair_pearson_correlation(pred_i, pred_j): 
    """
    Pearson Correlation between pair of output predictions 
    """
    mean_i = torch.mean(pred_i, dim=-1, keepdim=True)
    mean_j = torch.mean(pred_j, dim=-1, keepdim=True)
    delta_i = pred_i - mean_i
    delta_j = pred_j - mean_j 
    numerator = torch.sum(delta_i * delta_j, dim=-1) # [batch_size]
    norm_i = torch.sqrt(torch.sum(torch.square(delta_i),dim=-1)) # [batch_size]
    norm_j = torch.sqrt(torch.sum(torch.square(delta_j),dim=-1)) # [batch_size]
    output = numerator / norm_i / norm_j
    return output.mean()

def pearson_correlation(preds): 
    """
    """
    assert(len(preds) >= 2)
    num_pairs = 0 
    for i, pred_i in enumerate(preds): 
        for j, pred_j in enumerate(preds): 
            if i == j: 
                continue
            num_pairs += 1 
            output = pair_pearson_correlation(pred_i, pred_j)
    return output / num_pairs

def pair_cosine(pred_i, pred_j): 
    """
    """
    return torch.mean(torch.cosine_similarity(pred_i, pred_j, dim=-1)) 

def cosine_similarity(preds): 
    assert(len(preds) >= 2)
    num_pairs = 0 
    for i, pred_i in enumerate(preds): 
        for j, pred_j in enumerate(preds): 
            if i == j: 
                continue
            num_pairs += 1 
            output = pair_cosine(pred_i, pred_j)
    return output / num_pairs


def test_metrics(): 
    y_true = [0,0,0,0,0,0,0]
    y_pred1 = [
            # spread equally 
            [[0, 0.33, 0.33, 0.33, 0., 0., 0., 0., 0., 0.], # -3.3561
                [0, 0., 0., 0., 0.33, 0.33, 0.33, 0., 0., 0.], 
                [0, 0., 0., 0., 0., 0., 0., 0.33, 0.33, 0.33]],

            [[0.4, 0.2, 0.2, 0.2, 0., 0., 0., 0., 0., 0.], # -6.36
                [0.4, 0., 0., 0., 0.2, 0.2, 0.2, 0., 0., 0.], 
                [0.4, 0., 0., 0., 0., 0., 0., 0.2, 0.2, 0.2]],

            # prefer more confident in non-true 
            [[0, 0.9, 0.1, 0.0, 0., 0., 0., 0., 0., 0.], # -0.5953
            [0, 0., 0., 0., 0.9, 0.1, 0.0, 0., 0., 0.], 
            [0, 0., 0., 0., 0., 0., 0., 0.9, 0.1, 0.0]], 

            [[0.4, 0.4, 0.2, 0.0, 0., 0., 0., 0., 0., 0.], # -4.82
            [0.4, 0., 0., 0., 0.4, 0.2, 0.0, 0., 0., 0.], 
            [0.4, 0., 0., 0., 0., 0., 0., 0.4, 0.2, 0.0]], 

            # non-diverse 
            [[0, 0.33, 0.33, 0.33, 0., 0., 0., 0., 0., 0.],  # -27.6511
                [0, 0.33, 0.33, 0.33, 0., 0., 0., 0., 0., 0.], 
                [0, 0.33, 0.33, 0.33, 0., 0., 0., 0., 0., 0.]], 
            
            # prefer even more confident 
            [[0, 1.0, 0.0, 0.0, 0., 0., 0., 0., 0., 0.], # 0 
                [0, 0., 0., 0., 1.0, 0.0, 0.0, 0., 0., 0.], 
                [0, 0., 0., 0., 0., 0., 0., 1.0, 0.0, 0.0]], 
            
            # partly non-diverse 
            [[0, 1.0, 0.0, 0.0, 0., 0., 0., 0., 0., 0.], # -13.12
                [0, 0., 0., 0., 1.0, 0.0, 0.0, 0., 0., 0.], 
                [0, 0., 0., 0., 1.0, 0., 0., 0.0, 0.0, 0.0]], 
            ]
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(np.asarray(y_pred1))
    y_pred = torch.unsqueeze(y_pred,dim=2)
    DM = DiversityMeasure()
    for i in range(y_true.shape[0]): 
        targets = y_true[i] # [1,]
        preds = [y_pred[i,0,:,:],y_pred[i,1,:,:],y_pred[i,2,:,:]]    
        results = DM.standard(preds, targets)
        print('--------')
        print(results)

    """"
    Test Scenario: 
    - Case_1: distributed equally through all non-true
    - Case_2: distributed equally through all non-true + Correct prediction 
    - Case_3: One dominated class in Non-true set 
    - Case_4: One dominated class in Non-true set + Correct prediction 
    - Case_5: Completely Overlapping, non-diverse 
    - Case_6: Partly Overlapping, Partly non-diverse   
    """
if __name__ == '__main__': 
    test_metrics()