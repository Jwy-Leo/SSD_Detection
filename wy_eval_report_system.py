import torch
import numpy as np
def main():
    batch_size = 16
    num_batch = 4
    categorys = 10
    fake_dataset = lambda batch_size : gen_pattern(batch_size)
    fake_dataloder = map(fake_dataset, [ batch_size for i in range(num_batch)])
    TotalReport = compose_analysis_items([
        Precision_report(),
        Recall_report(),
        class_imbalance_report(),
        class_imbalance_correction_report(),
        Detection_localization_report()
    ])
    report_template(fake_dataloder,TotalReport)
    pass
def report_template(dataloader, TotalReport, categorys = 10):
    for index, (images, locations, labels) in enumerate(dataloader):
        # Inputpaload = {"Images":images, "locations": locations,"HELLO":"1"}
        
        # Images = images, locations = locations, classification_predictions = predictions, classification_ground_truth = ground_truths
        
        
        predictions = []
        for location in locations:
            random_gen = torch.rand((location.shape[0],categorys))
            random_gen /= torch.norm(random_gen, 1, dim = 1, keepdim = True)
            predictions.append(random_gen)
        # ground_truths = [ torch.nn.functional.dropout(torch.ones((location.shape[0],categorys))*0.5) for location in locations ]
        Inputpaload = {"Images" : images, "locations" : locations, "classification_predictions" : predictions, "classification_ground_truth" : labels}

        TotalReport(**Inputpaload)
def gen_pattern(batch_size, CHW = (3, 224, 224), categorys = 10):
    shape_size = (batch_size, ) + CHW
    object_size_gen = lambda item_size : np.random.randint(item_size)+1
    images = torch.randn(*shape_size)
    container = []
    label_container = []
    for i in range(batch_size):
        # (x, y, w_ratio, h_ratio)
        num_anchor = object_size_gen(30)
        data = torch.rand(num_anchor, 4)
        container.append(data)
        label_container.append(torch.randint(high = categorys, size = (num_anchor, )))
        
    locations = container
    return images, locations, label_container
class compose_analysis_items(object):
    def __init__(self, detection_wy_report_analysis):
        self.reports = detection_wy_report_analysis
    def __call__(self,**args):
        for report in self.reports:
            report(**args)
    def clean_all_report(self):
        for report in self.reports:
            report.clean_report()
    def clean_specifical_report(self):
        pass
    def Access_report_name(self):
        pass
class faliure_case_analysis(object):
    def __init__(self):
        self.report_item = {
            "false_positive" : 0,
            "false_negative" : 0,
            "true_positive" : 0,
            "true_negative":0
        }
        pass
    def __call__(self, **args):
        pass
    def clean_report(self):
        for name in self.report.keys():
            self.report[name] = 0
    def Access_report_name(self):
        return self.report.keys()
class Precision_report(object):
    def __init__(self):
        self.false_positive = 0
        self.false_negative = 0
        self.true_positive = 0
    def __call__(self, **args):
        import pdb;pdb.set_trace()
        pass
    def clean_report(self):
        self.false_positive = 0
        self.false_negative = 0
        self.true_positive = 0
    def Access_report_name(self):
        return ["prediction","ground_truth"]
class Recall_report(object):
    def __init__(self):
        pass
    def __call__(self, **args):
        pass
    def clean_report(self):
        pass
    def Access_report_name(self):
        pass
class class_imbalance_report(object):
    def __init__(self):
        pass
    def __call__(self, **args):
        pass
class class_imbalance_correction_report(object):
    def __init__(self):
        pass
    def __call__(self, **args):
        pass
class Regression_scalar_report(object):
    def __init__(self):
        pass
    def __call__(self, **args):
        pass
class Detection_localization_report(object):
    def __init__(self):
        self.loc_w_dist = Regression_scalar_report()
        self.loc_h_dist = Regression_scalar_report()
        self.ratio_w_dist = Regression_scalar_report()
        self.ratio_h_dist = Regression_scalar_report()
    def __call__(self, **args):
        pass

if __name__ == "__main__":
    main()