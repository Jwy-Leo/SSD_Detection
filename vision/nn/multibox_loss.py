import torch.nn as nn
import torch.nn.functional as F
import torch


from ..utils import box_utils


class MultiboxLoss(nn.Module):
    def __init__(self, priors, iou_threshold, neg_pos_ratio,
                 center_variance, size_variance, device):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiboxLoss, self).__init__()
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors.to(device)

    def forward(self, confidence, predicted_locations, labels, gt_locations, weighted_vector=None):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        

        query_table = (torch.isinf(gt_locations[:,:,2]) + torch.isinf(gt_locations[:,:,3])) ==0
        gt_locations = gt_locations[query_table,:]
        predicted_locations = predicted_locations[query_table,:]
        confidence = confidence[query_table,:]
        labels = labels[query_table]

        num_classes = confidence.size(1)

        with torch.no_grad():
            loss = -1 * F.log_softmax(confidence, dim=1)[:, 0]
            loss = loss.unsqueeze(dim=0)
            labels = labels.unsqueeze(dim=0)
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)
            mask = mask.squeeze(dim=0)
            labels = labels.squeeze(dim=0)

        confidence = confidence[mask, :]
        if int(torch.sum(confidence).data.cpu()) == 0:
            print("Only have background sample")
            return 0 * torch.sum(confidence), 0 * torch.sum(confidence)
        else:
            final_label = labels[mask]
            if final_label.type() == "torch.FloatTensor":
                final_label = torch.LongTensor(final_label).cuda()
            elif final_label.type() == "torch.cuda.FloatTensor":
                final_label = final_label.data.cpu().numpy()
                final_label = torch.LongTensor(final_label).cuda()
            elif final_label.type() == "torch.cuda.LongTensor":
                pass
            else:
                print(final_label.type())
            if weighted_vector is not None:
                classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), final_label, weighted_vector, size_average=False)
            else:
                classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), final_label, size_average=False)
        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
        gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
        num_pos = gt_locations.size(0)
        if num_pos !=0:
            return smooth_l1_loss/num_pos, classification_loss/num_pos
        else:
            print("L1_loss{}, classification{}".format(smooth_l1_loss, classification_loss))
            return smooth_l1_loss * 0, classification_loss* 0
        
        # num_classes = confidence.size(2)

        # TODO
        # Using non-maximum suppresion to avoid anchor duplicated
        # picked_index = []
        # for class_index in range(1, confidence.size(1)):
            # probs = confidence[..., class_index].unsqueeze(dim=2)
            # if probs.size(1) == 0:
            #     continue
            # box_probs = torch.cat([predicted_locations, probs], dim=2)
            # _picked_index = box_utils.hard_nms_ret_index(box_probs,iou_threshold = 0.7)
            # picked_index.append(_picked_index)
        
        # picked_index = torch.cat(picked_index)
        # confidence = confidence[picked_index,:]
        # loss = loss[picked_index,:]
        # labels = labels[picked_index,:]

        # with torch.no_grad():
        #     # derived from cross_entropy=sum(log(p))
        #     import pdb;pdb.set_trace()
        #     loss = -1 * F.log_softmax(confidence, dim=2)[:, :, 0]
        #     mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)

        # confidence = confidence[mask, :]
        # if int(torch.sum(confidence).data.cpu()) == 0:
        #     print("Only have background sample")
        #     return 0 * torch.sum(confidence), 0 * torch.sum(confidence)
        # else:
        #     final_label = labels[mask]
        #     if final_label.type() == "torch.FloatTensor":
        #         final_label = torch.LongTensor(final_label).cuda()
        #     elif final_label.type() == "torch.cuda.FloatTensor":
        #         final_label = final_label.data.cpu().numpy()
        #         final_label = torch.LongTensor(final_label).cuda()
        #     elif final_label.type() == "torch.cuda.LongTensor":
        #         pass
        #     else:
        #         print(final_label.type())
        #     if weighted_vector is not None:
        #         classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), final_label, weighted_vector, size_average=False)
        #     else:
        #         classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), final_label, size_average=False)
        # pos_mask = labels > 0
        # predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
        # gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
        
        # # drop_table = (torch.isinf(gt_locations[:,2]) + torch.isinf(gt_locations[:,3])) ==0
        # # gt_locations = gt_locations[drop_table,:]
        # # predicted_locations = predicted_locations[drop_table,:]

        # smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
        # num_pos = gt_locations.size(0)
        # if num_pos !=0:
        #     return smooth_l1_loss/num_pos, classification_loss/num_pos
        # else:
        #     # smooth_l1_loss * 0, classification_loss
        #     print("L1_loss{}, classification{}".format(smooth_l1_loss, classification_loss))
        #     return smooth_l1_loss * 0, classification_loss* 0
    def bbox_regression_losss(self,gt_location_x1y1x2y2,prediction_x1y1x2y2):
        #caculate_f = F.smooth_l1_loss
        #prediction
        #center_loss = 
        #w_ratio = torch.log()
        #return
        pass
    def _cornerformat_convert_centerformat(self):
        pass
