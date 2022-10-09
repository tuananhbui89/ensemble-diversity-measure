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
        elif metric == 'pearson_distance': 
            return 1 - pearson_correlation(preds)
        elif metric == 'cosine_distance': 
            return 1 - cosine_similarity(preds)
        elif metric == 'log_det_nonmax': 
            return log_det(y_true=targets, y_pred=preds).mean()
    
    def standard(self, preds, targets): 
        """
        Higher score means higher diversity 
        """
        metrics = ['disagreement', 
                'norm_disagreement',
                'double_fault_measure', 
                'pearson_distance',
                'cosine_distance',
                'log_det_nonmax']
                
        output = dict()
        for m in metrics: 
            output[m] = self.one_metric(preds, targets, m)
        return output

def pair_disagreement(pred_i, pred_j): 
    """
    Disagreement between a pair of preds 
    Description about the metric: 
        Higher score means higher disagreement --> higher diversity 
        Highest = 1 --> totally disagreement 
        Lowest = 0 --> totally agreement 
    """
    output = torch.ne(torch.argmax(pred_i, dim=-1), torch.argmax(pred_j, dim=-1))
    return torch.sum(output.float())/output.shape[0]

def disagreement(preds): 
    """
    Ref: http://ceur-ws.org/Vol-2916/paper_8.pdf 
    Disagree of pair classifiers: D_{i,j} = N^{\tilde{y_i} \neq \tilde{y_j}} / N 
    \tilde{y_i}: predicted label of classifier i 
    \tilde{y_j}: predicted label of classifier j 
    Args: 
        - preds: list of prediction [pred, ... ,pred], each pred has shape [batch_size, num_classes]
    Description about the metric: 
        Higher score means higher disagreement --> higher diversity 
        Highest = 1 --> totally disagreement 
        Lowest = 0 --> totally agreement 

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
    Description about the metric: 
        Higher score means higher disagreement --> higher diversity 
        Highest = 1 --> totally disagreement 
        Lowest = 0 --> totally agreement   
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
    """
    Rate when both members make wrong predictions 
    Higher score means more incorrect predictions --> Higher diversity 
    """
    output1 = torch.eq(targets, torch.argmax(pred_i, dim=-1)).float() # [0,1]
    output2 = torch.eq(targets, torch.argmax(pred_j, dim=-1)).float() # [0,1]
    output = (output1 + output2) < 2 
    return torch.mean(output.float())


def double_fault_measure(preds, targets): 
    """"
    DF_{i.j} = N^{00}_{i,j}/N 
    where N^{00}_{i,j} is number of samples that both members make wrong predictions 
    Higher score means more incorrect predictions --> Higher diversity 
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
    Lower score means higher diversity 
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
    Lower score means higher diversity 
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
    Lower score means higher diversity 
    """
    return torch.mean(torch.cosine_similarity(pred_i, pred_j, dim=-1)) 

def cosine_similarity(preds): 
    """
    Lower score means higher diversity 
    """
    assert(len(preds) >= 2)
    num_pairs = 0 
    for i, pred_i in enumerate(preds): 
        for j, pred_j in enumerate(preds): 
            if i == j: 
                continue
            num_pairs += 1 
            output = pair_cosine(pred_i, pred_j)
    return output / num_pairs


def log_det(y_true, y_pred):
    """
    Args: 
        y_true: categorial format 
        y_pred: prediction (probability), in list format  
    """
    assert(type(y_pred) is list)
    assert(y_true.shape[0] == y_pred[0].shape[0])
    num_models = len(y_pred)

    """
    Step 1: Find non-true mask, using pytorch masked_select 
        Apply separately for each pair [pred, label]  
        Mask: [batch_size, num_classes]: Notlabel --> True, Label --> False 
        Pred: [batch_size, num_classes]
        Return: [batch_size, num_classes-1]

    """
    batch_size, num_classes = y_pred[0].shape
    mask_non_true = (1 - F.one_hot(y_true, num_classes=num_classes)).bool()
    flat_mask_non_true = torch.reshape(mask_non_true, [batch_size*num_classes,])

    masked_preds = []
    for pred in y_pred: 
        flat_pred = torch.reshape(pred, [batch_size*num_classes,])
        _masked_pred = torch.masked_select(flat_pred, flat_mask_non_true)
        assert(_masked_pred.shape[0] == batch_size * (num_classes-1))
        _masked_pred = torch.reshape(_masked_pred, [batch_size, num_classes-1])
        masked_preds.append(_masked_pred)

    masked_preds = torch.stack(masked_preds, dim=1).to(device) # [batch_size, num_models, num_classes-1]
    
    # DONOT USE Normalized. Make NaN problem when calculating logdet  
    # masked_preds = masked_preds / torch.norm(masked_preds, p=2, dim=2, keepdim=True)
    # masked_preds = masked_preds / torch.sum(masked_preds, dim=2, keepdim=True)
    # print(masked_preds)
    
    # Step 2: Calculate LogDet 
    matrix = torch.matmul(masked_preds, torch.transpose(masked_preds, dim0=1, dim1=2)) # [batch_size, num_models, num_models]
    assert(list(matrix.shape) == [batch_size, num_models, num_models], 
            "shape of matrix should be [batch_size, num_models, num_models] but get {}".format(matrix.shape)) 

    all_log_det = torch.logdet(matrix + det_offset * torch.unsqueeze(torch.eye(num_models), dim=0).to(device)) # [batch_size,]
    return all_log_det

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