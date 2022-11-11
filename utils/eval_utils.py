import torch
import numpy as np


def eval_worker(eval_dict, logger):
    model = eval_dict["model"]
    dataloader = eval_dict["dataloader"]
    best_target_acc = eval_dict["best_target_acc"]
    dataset = eval_dict["dataset"]
    device = eval_dict["device"]
    criterion = eval_dict["criterion"]
    # TODO when do eval, should still CE be used?
    epoch = eval_dict["epoch"]
    best_target_acc_epoch = eval_dict["best_target_acc_epoch"]
    num_class = eval_dict["num_class"]
    if "source_flag" in eval_dict:
        source_flag = True
    else:
        source_flag = False
    
    if "cls_eval" in eval_dict:
        cls_eval = eval_dict["cls_eval"]
    else:
        cls_eval = False

    logger.info(f"Current eval on: {dataset} {eval_dict['dataset_name']}")
    loss_total = 0
    correct_total = 0
    data_total = 0
    acc_class = torch.zeros(10, 1)
    acc_to_class = torch.zeros(10, 1)
    acc_to_all_class = torch.zeros(10, 10)

    class_acc = np.zeros((num_class, 3))
    mean_correct = []

    for batch_idx, (data, label) in enumerate(dataloader):
        data = data.to(device=device)
        label = label.to(device=device).long()
        if source_flag:
            output = model(data)
        else:
            pred1, pred2 = model(data)
            output = (pred1 + pred2) / 2

        loss = criterion(output, label)
        _, pred = torch.max(output, 1)

        if source_flag or cls_eval:
            acc = pred == label
            for j in np.unique(label.cpu()):
                classacc = pred[label == j].eq(label[label == j].long().data).cpu().sum()
                class_acc[j, 0] += classacc.item() / float(data[label == j].size()[0])
                class_acc[j, 1] += 1

        correct = pred.eq(label.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(data.size()[0]))

        loss_total += loss.item() * data.size(0)
        correct_total += torch.sum(pred == label)
        data_total += data.size(0)

    pred_loss = loss_total / data_total
    pred_acc = correct_total.double() / data_total

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc_ = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    if pred_acc > best_target_acc:
        best_target_acc = pred_acc
        best_target_acc_epoch = epoch
    logger.info(
        f'On dataset {dataset} :{epoch} [overall_acc: {pred_acc} Best Tar Acc: {best_target_acc} on Source Train Epoch {best_target_acc_epoch}]')

    if source_flag or cls_eval:
        
        logger.info(f"Cls-wise eval: {class_acc[:, 2]}")
        logger.info(f"compared eval: {instance_acc} and avg: {class_acc_}")

    result = {
        "dataset": dataset,
        "epoch": epoch,
        "best_target_acc": best_target_acc,
        "best_target_acc_epoch": best_target_acc_epoch,
        "cur_target_acc": pred_acc
    }
    return result
