import torch


def eval_worker(eval_dict, logger):
    model = eval_dict["model"]
    dataloader = eval_dict["dataloader"]
    best_target_acc = eval_dict["best_target_acc"]
    dataset = eval_dict["dataset"]
    device = eval_dict["device"]
    criterion = eval_dict["criterion"]
    epoch = eval_dict["epoch"]
    best_target_acc_epoch = eval_dict["best_target_acc_epoch"]
    num_class = eval_dict["num_class"]
    if "source_flag" in eval_dict:
        source_flag = True
    logger.info(f"Current eval on: {dataset} {eval_dict['dataset_name']}")
    loss_total = 0
    correct_total = 0
    data_total = 0
    acc_class = torch.zeros(10, 1)
    acc_to_class = torch.zeros(10, 1)
    acc_to_all_class = torch.zeros(10, 10)

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

        if source_flag:
            acc = pred == label
            for j in range(num_class):
                label_j_list = (label == j)
                acc_class[j] += (pred[acc] == j).sum().cpu().float()
                acc_to_class[j] += label_j_list.sum().cpu().float()
                for k in range(num_class):
                    acc_to_all_class[j, k] += (pred[label_j_list]
                                               == k).sum().cpu().float()

        loss_total += loss.item() * data.size(0)
        correct_total += torch.sum(pred == label)
        data_total += data.size(0)

    pred_loss = loss_total / data_total
    pred_acc = correct_total.double() / data_total

    if pred_acc > best_target_acc:
        best_target_acc = pred_acc
        best_target_acc_epoch = epoch
    logger.info(
        f'On dataset {dataset} :{epoch} [overall_acc: {pred_acc} Best Tar Acc: {best_target_acc} on Source Train Epoch {best_target_acc_epoch}]')

    result = {
        "dataset": dataset,
        "epoch": epoch,
        "best_target_acc": best_target_acc,
        "best_target_acc_epoch": best_target_acc_epoch
    }
    return result
