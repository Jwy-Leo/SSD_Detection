import numpy as np
def main():
    # random_sampling(template = "hello", HIHI = "KK")
    random_sampling("hello","KK")
def sampling_method_template():
    pass

def random_sampling(*argc,**argv):
    '''
    Input signal
    unlabeled_list
    query_item
    '''
    
    if argv.keys()!=0:
        unlabeled_list = argv['unlabeld_list']
        query_item = argv['query_item']
    else:
        if "warning_flag" not in dir(random_sampling):
            random_sampling.warning_flag = True
            print("Warning input [0] unlabeld_list, [1] query_item")
        unlabeled_list = argc[0]
        query_item = argc[1]
    query_index = np.random.choice(unlabeled_list,query_item,replace = False)

    return query_index
def sequential_sampling(*argc, **argv):
    '''
    Input signal
    unlabeled_list
    query_item
    '''
    if argv.keys()!=0:
        unlabeled_list = argv['unlabeld_list']
        query_item = argv['query_item']
    else:
        if "warning_flag" not in dir(random_sampling):
            random_sampling.warning_flag = True
            print("Warning input [0] unlabeld_list, [1] query_item")
        unlabeled_list = argc[0]
        query_item = argc[1]
    query_index = unlabeled_list[:query_item]

    return query_index
def model_inference(imgs_list, net):

    max_num = 50
    confidences = []
    for index in range( len(imgs_list)//max_num + 1 ):
        with torch.no_grad():
            begin_pointer = index * max_num
            end_pointer = min( (index+1) * max_num, len(imgs_list))
            sub_batch = torch.stack(imgs_list[begin_pointer:end_pointer]).cuda()
            _confidence, locations = net(sub_batch)
            confidences.append(_confidence.data.cpu())
    confidences = torch.cat(confidences, 0)

    return confidences
def uncertainty_sampling(*argc, **argv):
    '''
    Input signal
    unlabeled_list
    query_item
    imgs_list
    net
    '''
    if argv.keys()!=0:
        unlabeled_list = argv['unlabeld_list']
        query_item = argv['query_item']
        imgs_list = argv['imgs_list']
        net = argv['net']
    else:
        if "warning_flag" not in dir(random_sampling):
            random_sampling.warning_flag = True
            print("Warning input [0] unlabeld_list, [1] query_item, [2] imgs_list, [3] net")
        unlabeled_list = argc[0]
        query_item = argc[1]
        imgs_list = argc[2]
        net = argc[3]
    
    
    confidences = model_inference(imgs_list,net)    
    probability = torch.softmax(confidences, 2)
    entropy = torch.sum(probability * torch.log(probability) * -1, 2)
    maximum = torch.max(entropy, 1)[0]
    criteria = maximum
    query_index = torch.argsort( -1 * criteria)[:query_item].tolist()

    return query_index
def uncertainty_modify_sampling(*argc, **argv):
    '''
    Input signal
    unlabeled_list
    query_item
    imgs_list
    net
    '''
    if argv.keys()!=0:
        unlabeled_list = argv['unlabeld_list']
        query_item = argv['query_item']
        imgs_list = argv['imgs_list']
        net = argv['net']
    else:
        if "warning_flag" not in dir(random_sampling):
            random_sampling.warning_flag = True
            print("Warning input [0] unlabeld_list, [1] query_item, [2] imgs_list, [3] net")
        unlabeled_list = argc[0]
        query_item = argc[1]
        imgs_list = argc[2]
        net = argc[3]

    confidences = model_inference(imgs_list,net)    
    probability = torch.softmax(confidences, 2)
    entropy = torch.sum(probability * torch.log(probability) * -1, 2)
    mean = torch.mean(entropy, 1)
    stddev = torch.std(entropy, 1)
    criteria = mean * stddev / (mean+stddev)
    query_index = torch.argsort( -1 * criteria)[:query_item].tolist()

    return query_index

def diversity_sampling(*argc, **argv):
    import pdb;pdb.set_trace()
    pass
def balance_feature_sampling(*argc, **argv):
    import pdb;pdb.set_trace()
    pass
if __name__ == "__main__":
    main()
