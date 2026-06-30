import argparse
import contextlib
import json
import os
import clip
import torch
import random
import time
import numpy as np
import pandas as pd
from torch.optim import lr_scheduler
from my_loss import pLoss_all_fidelity
from legal_states import list_legal_states
from coop import CustomCLIP
from label_schema import EVAL_LABELS_MIN100_GLOBAL, QUESTION_COLS
from metrics_utils import compute_per_attribute_metrics
from utils import set_dataset, _preprocess2, _preprocess3, save_metrics_csv_and_plot, \
    load_checkpoint_if_any, resolve_split_csv_paths


ATTRS = QUESTION_COLS
num_levels = 4
num_attr = len(ATTRS)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
AMP_ENABLED = torch.cuda.is_available()
AMP_DTYPE = torch.bfloat16 if AMP_ENABLED and torch.cuda.is_bf16_supported() else torch.float16
AMP_MODE_NAME = "bf16" if AMP_DTYPE == torch.bfloat16 else "fp16"

stage = 1
BEST_SELECTOR_METRIC = "val_auc_all"
TRACKED_METRICS = ["acc", "auc", "qwk", "recall"]
MAX_BEST_CKPTS = 3


def amp_autocast():
    if not AMP_ENABLED:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=AMP_DTYPE)


def _is_better_epoch(best_result, val_metrics):
    current = (float(val_metrics["auc"]["all"]), float(val_metrics["acc"]["all"]))
    if not best_result:
        return True
    best = (float(best_result["val_auc_all"]), float(best_result["val_acc_all"]))
    return current > best


def _save_best_epoch_metrics(
    output_path,
    best_epoch,
    display_labels,
    val_counts,
    val_metrics,
    test_counts,
    test_metrics,
):
    rows = []
    for label in list(display_labels) + ["all"]:
        rows.append({
            "best_epoch": int(best_epoch),
            "selection_metric": BEST_SELECTOR_METRIC,
            "selection_value": float(val_metrics["auc"]["all"]),
            "label": label,
            "val_image_count": int(val_counts[label]),
            "val_acc": float(val_metrics["acc"][label]),
            "val_auc": float(val_metrics["auc"][label]),
            "val_qwk": float(val_metrics["qwk"][label]),
            "val_recall": float(val_metrics["recall"][label]),
            "test_image_count": int(test_counts[label]),
            "test_acc": float(test_metrics["acc"][label]),
            "test_auc": float(test_metrics["auc"][label]),
            "test_qwk": float(test_metrics["qwk"][label]),
            "test_recall": float(test_metrics["recall"][label]),
        })

    out_csv = os.path.join(output_path, "best_epoch_metrics.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"[Best] saved {out_csv}")


def _load_existing_best_result(output_path):
    path = os.path.join(output_path, "best_epoch_metrics.csv")
    if not os.path.exists(path):
        return {}, 0
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[Best] failed to read existing metrics {path}: {exc}")
        return {}, 0

    if "label" not in df.columns:
        return {}, 0
    row = df[df["label"] == "all"]
    if row.empty:
        return {}, 0
    row = row.iloc[0]
    best_epoch = int(row["best_epoch"])
    best_result = {
        "epoch": best_epoch,
        "val_auc_all": float(row["val_auc"]),
        "val_acc_all": float(row["val_acc"]),
        "test_auc_all": float(row["test_auc"]),
        "test_acc_all": float(row["test_acc"]),
    }
    print(
        f"[Best] restored existing best epoch {best_epoch}: "
        f"val_auc_all={best_result['val_auc_all']:.4f}"
    )
    return best_result, best_epoch


def _best_ckpt_manifest_path(ckpt_path):
    return os.path.join(ckpt_path, "best_checkpoints.json")


def _best_ckpt_filename(epoch):
    return f"MORE_IQ_best_{epoch:05d}.pt"


def _load_best_ckpt_manifest(ckpt_path):
    manifest_path = _best_ckpt_manifest_path(ckpt_path)
    if not os.path.exists(manifest_path):
        return []
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except Exception as exc:
        print(f"[ckpt] Failed to load manifest {manifest_path}: {exc}")
        return []

    cleaned = []
    for record in records:
        path = record.get("path")
        if path and os.path.exists(path):
            cleaned.append(record)
    return cleaned


def _save_best_ckpt_manifest(ckpt_path, records):
    manifest_path = _best_ckpt_manifest_path(ckpt_path)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=True)


def _save_last_checkpoint(ckpt_path, ckpt_payload):
    last_path = os.path.join(ckpt_path, "MORE_IQ-last.pt")
    torch.save(ckpt_payload, last_path)
    print(f"[ckpt] updated last checkpoint: {last_path}")


def _update_best_checkpoints(ckpt_path, epoch, val_metrics, ckpt_payload):
    candidate = {
        "epoch": int(epoch),
        "val_auc_all": float(val_metrics["auc"]["all"]),
        "val_acc_all": float(val_metrics["acc"]["all"]),
        "path": os.path.join(ckpt_path, _best_ckpt_filename(epoch)),
    }
    records = [r for r in _load_best_ckpt_manifest(ckpt_path) if int(r.get("epoch", -1)) != int(epoch)]
    combined = records + [candidate]
    combined.sort(
        key=lambda r: (float(r["val_auc_all"]), float(r["val_acc_all"]), int(r["epoch"])),
        reverse=True,
    )
    keep = combined[:MAX_BEST_CKPTS]
    keep_paths = {r["path"] for r in keep}

    if candidate["path"] in keep_paths:
        torch.save(ckpt_payload, candidate["path"])
        print(
            f"[ckpt] kept best checkpoint: {candidate['path']} "
            f"(val_auc_all={candidate['val_auc_all']:.4f})"
        )
    else:
        print(
            f"[ckpt] skipped checkpoint at epoch {epoch} "
            f"(val_auc_all={candidate['val_auc_all']:.4f})"
        )

    for record in combined:
        path = record["path"]
        if path not in keep_paths and os.path.exists(path):
            os.remove(path)
            print(f"[ckpt] removed checkpoint: {path}")

    _save_best_ckpt_manifest(ckpt_path, keep)

def parse_args():
    parser = argparse.ArgumentParser(description="Train PrISM-IQA on one data split.")
    parser.add_argument("--split", default=os.environ.get("MORE_IQ_SPLIT", "split0"))
    parser.add_argument("--csv-root", default="./csv_file_merged")
    parser.add_argument("--image-path", default="/mnt/bn/public-vqa1/SPAQ/SPAQ_dataset/TestImage")
    parser.add_argument("--output-root", default="./output")
    return parser.parse_args()


def main(args):
    num_epoch = 50
    effective_batch_size = 50
    train_micro_batch_size = 25
    grad_accum_steps = effective_batch_size // train_micro_batch_size
    eval_batch_size = 32
    if effective_batch_size % train_micro_batch_size != 0:
        raise ValueError("effective_batch_size must be divisible by train_micro_batch_size")
    num_patch_train = 5
    num_patch_test = 9
    num_workers = 8

    # learning rate
    if stage == 1:
        lr_visual = 5e-6
    else:
        lr_visual = 5e-6

    lr_prompt = 5e-4
    lr_scale = 1e-4
    lr_delta = 1e-3

    # FFL loss
    # alpha = torch.tensor([0.4, 0.5, 2.2, 0.5])
    # alpha = alpha.to(device)
    alpha = None
    gamma = 1

    # CoOp
    n_ctx = 4
    specific = True
    num_binary = num_levels - 1

    # file path
    split = args.split
    image_path = args.image_path  # path of image
    train_csv, val_csv, test_csv, split_dir = resolve_split_csv_paths(args.csv_root, split)
    ignored_label_columns = [label for label in ATTRS if label not in set(EVAL_LABELS_MIN100_GLOBAL)]
    ckpt_path = os.path.join(args.output_root, 'ckpt', split)
    output_path = os.path.join(args.output_root, 'results', split)
    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    print(f"Using split: {split} ({split_dir})")
    print(
        f"Train effective batch size: {effective_batch_size} "
        f"(micro-batch={train_micro_batch_size}, grad_accum_steps={grad_accum_steps})"
    )
    print(f"AMP enabled: {AMP_ENABLED} ({AMP_MODE_NAME if AMP_ENABLED else 'off'})")

    # -----------------------------------
    # define the model
    # -----------------------------------
    clip_model, preprocess = clip.load("ViT-L/14", device='cpu')
    clip_model.float()   # float16 -> float32
    model = CustomCLIP(clip_model, ATTRS, n_ctx=n_ctx, n_binary=num_binary, specific=specific).to(device)
    for p in model.text_encoder.parameters():
        p.requires_grad = False
    model.logit_scale.requires_grad = False # ensure logit_scale requires grad
    print("Model prepared")

    # -----------------------------------
    # optimizer
    # -----------------------------------
    prompt_params = model.prompt_learner.parameters()
    visual_params = model.image_encoder.parameters()
    scale_params = [model.logit_scale]
    #delta_params = [model.delta]
    #base_params = [model.base]
    tau_params = [model.tau1, model.tau_gap_raw]

    if stage == 1:
        optimizer = torch.optim.AdamW([
            {'params': prompt_params, 'lr': lr_prompt, 'weight_decay': 0.001},
            {'params': visual_params, 'lr': lr_visual, 'weight_decay': 0.001},
            # {'params': tau_params, 'lr': lr_delta, 'weight_decay': 0.0},
            # {'params': scale_params, 'lr': lr_scale, 'weight_decay': 0.001},
            # {'params': delta_params, 'lr': lr_delta, 'weight_decay': 0.0},
            # {'params': base_params, 'lr': lr_delta, 'weight_decay': 0.0},
        ])
    else:
        optimizer = torch.optim.AdamW([
            #{'params': prompt_params, 'lr': lr_prompt, 'weight_decay': 0.001},
            #{'params': visual_params, 'lr': lr_visual, 'weight_decay': 0.001},
            {'params': tau_params, 'lr': lr_delta, 'weight_decay': 0.0},
            # {'params': scale_params, 'lr': lr_scale, 'weight_decay': 0.001},
            # {'params': delta_params, 'lr': lr_delta, 'weight_decay': 0.0},
            # {'params': base_params, 'lr': lr_delta, 'weight_decay': 0.0},
        ])


    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    # -----------------------------------
    # define the loss
    # -----------------------------------
    criterion = pLoss_all_fidelity(states=list_legal_states(), alpha=alpha, gamma=gamma).to(device)

    # -----------------------------------
    # define the dataset
    # -----------------------------------
    train_loader = set_dataset(
        train_csv, train_micro_batch_size, image_path, num_workers, preprocess3,
        num_patch_train, num_attr, False, label_columns=ATTRS,
        ignored_label_columns=ignored_label_columns
    )
    val_loader = set_dataset(
        val_csv, eval_batch_size, image_path, num_workers, preprocess2,
        num_patch_test, num_attr, True, label_columns=ATTRS,
        ignored_label_columns=ignored_label_columns
    )
    test_loader = set_dataset(
        test_csv, eval_batch_size, image_path, num_workers, preprocess2,
        num_patch_test, num_attr, True, label_columns=ATTRS,
        ignored_label_columns=ignored_label_columns
    )

    start_epoch = load_checkpoint_if_any(ckpt_path, model, optimizer, scheduler, device)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED and AMP_DTYPE == torch.float16)

    train_loss = []
    best_result, best_epoch = _load_existing_best_result(output_path)
    categories = ATTRS + ["all"]
    metrics_with_categories = [
        f"{phase}_{metric_name}"
        for phase in ("val", "test")
        for metric_name in TRACKED_METRICS
    ]
    hist = {
        "train_loss": [],
        **{m: {c: [] for c in categories} for m in metrics_with_categories}
    }

    print("Start training!")
    for epoch in range(start_epoch, num_epoch):
        print(f"scale parameters: {model.logit_scale.exp()}")
        #print(f"delta parameters: {model.delta.exp()}")
        print(f"tau parameters: {model.tau1}")
        print(f"tau_gap_raw parameters: {model.tau_gap_raw}")
        best_result, best_epoch = train_single_epoch(model, best_result, best_epoch,
                                                     train_loader, val_loader, test_loader,
                                                     epoch, start_epoch, train_loss,
                                                     ckpt_path, optimizer, scheduler,
                                                     criterion, hist, output_path, grad_accum_steps, scaler)
        scheduler.step()

    save_metrics_csv_and_plot(hist, output_path, ATTRS)
    done_marker = os.path.join(output_path, "training_done.txt")
    with open(done_marker, "w", encoding="utf-8") as f:
        f.write(f"completed_epochs={num_epoch}\n")
    print(f"[Done] wrote {done_marker}")


def do_batch(model, x):
    batch_size = x.size(0)
    num_patch = x.size(1)
    x = x.view(-1, x.size(2), x.size(3), x.size(4))
    logits_per_image = model.forward(x)
    logits_per_image = logits_per_image.view(batch_size, num_patch, -1)
    logits_per_image = logits_per_image.mean(1)
    return logits_per_image


def train_single_epoch(model, best_result, best_epoch, train_loader, val_loader, test_loader, epoch, start_epoch,
                       train_loss, ckpt_path, optimizer, scheduler, criterion, hist, output_path,
                       grad_accum_steps, scaler):
    start_time = time.time()
    beta = 0.9
    running_loss = train_loss[-1] if epoch > start_epoch else 0.0
    running_duration = 0.0
    num_steps_per_epoch = 10
    local_counter = epoch * num_steps_per_epoch + 1
    epoch_losses = []

    # model.eval()
    model.train()

    print('Learning rate:')
    print(optimizer.state_dict()['param_groups'][0]['lr'])
    if optimizer.state_dict()['param_groups'][0]['lr'] == 0:
        scheduler.step()
        print(optimizer.state_dict()['param_groups'][0]['lr'])

    num_sample_per_task = []
    optimizer.zero_grad()

    for dataset_idx, sample_batched in enumerate(train_loader, 0):

        x, all_node_gt = sample_batched['I'], sample_batched['all_node']
        observed_mask = sample_batched['observed_mask']
        x = x.to(device)
        all_node_gt = all_node_gt.to(device)   # （B, A, L）
        observed_mask = observed_mask.to(device)
        num_sample_per_task.append(x.size(0))

        with amp_autocast():
            logits_per_image = do_batch(model, x)
        logits_per_image = logits_per_image.float()
        all_loss, pMargin = criterion(logits_per_image, all_node_gt, observed_mask)

        loss_for_backward = all_loss / grad_accum_steps
        if scaler.is_enabled():
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()
        should_step = ((dataset_idx + 1) % grad_accum_steps == 0) or ((dataset_idx + 1) == len(train_loader))
        if should_step:
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        # statistics
        running_loss = beta * running_loss + (1 - beta) * all_loss.data.item()
        loss_corrected = running_loss / (1 - beta ** local_counter)

        current_time = time.time()
        duration = current_time - start_time
        running_duration = beta * running_duration + (1 - beta) * duration
        duration_corrected = running_duration / (1 - beta ** local_counter)
        examples_per_sec = x.size(0) / duration_corrected
        format_str = ('(E:%d, S:%d ) [Loss = %.4f] (%.1f samples/sec; %.3f '
                      'sec/batch)')
        print(format_str % (epoch, dataset_idx + 1, loss_corrected,
                            examples_per_sec, duration_corrected))

        local_counter += 1
        start_time = time.time()

        train_loss.append(loss_corrected)
        epoch_losses.append(all_loss.data.item())
        # break

    val_metrics, count_val, val_display_labels = (
        eval(model, criterion, val_loader, phase='val', dataset='oppo', epoch=epoch, output_dir=output_path))
    test_metrics, count_test, _ = (
        eval(model, criterion, test_loader, phase='test', dataset='oppo', epoch=epoch, output_dir=output_path))

    if _is_better_epoch(best_result, val_metrics):
        best_epoch = epoch
        best_result = {
            "epoch": int(epoch),
            "val_auc_all": float(val_metrics["auc"]["all"]),
            "val_acc_all": float(val_metrics["acc"]["all"]),
            "test_auc_all": float(test_metrics["auc"]["all"]),
            "test_acc_all": float(test_metrics["acc"]["all"]),
        }
        _save_best_epoch_metrics(
            output_path,
            best_epoch,
            val_display_labels,
            count_val,
            val_metrics,
            count_test,
            test_metrics,
        )
        print(
            f"[Best on val] epoch {best_epoch}: "
            f"val_auc_all={val_metrics['auc']['all']:.4f}, test_auc_all={test_metrics['auc']['all']:.4f}"
        )

    if hist is not None and len(epoch_losses) > 0:
        hist["train_loss"].append(float(np.mean(epoch_losses)))

    if hist is not None:
        for k in ATTRS + ["all"]:
            for metric_name in TRACKED_METRICS:
                hist[f"val_{metric_name}"][k].append(float(val_metrics[metric_name][k]))
                hist[f"test_{metric_name}"][k].append(float(test_metrics[metric_name][k]))

    ckpt_payload = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        #'optimizer_state_dict': optimizer.state_dict(),
    }
    _save_last_checkpoint(ckpt_path, ckpt_payload)
    _update_best_checkpoints(ckpt_path, epoch, val_metrics, ckpt_payload)

    return best_result, best_epoch


def eval(model, criterion, loader, phase, dataset, epoch, output_dir):
    model.eval()

    csv_rows = []
    pred_all = []
    gt_all = []
    prob_all = []
    observed_all = []

    with torch.no_grad():
        for step, sample_batched in enumerate(loader, 0):
            x, filenames, all_node = sample_batched['I'], sample_batched['filename'], sample_batched['all_node']
            observed_mask = sample_batched['observed_mask']
            x = x.to(device)
            all_node_gt = all_node.to(device)
            observed_mask = observed_mask.to(device)
            with amp_autocast():
                logits_per_image = do_batch(model, x)
            logits_per_image = logits_per_image.float()
            p_ord = criterion.crf(logits_per_image)
            pred_mapped = criterion.map_decode_levels(logits_per_image)  # (B, A) 0..3
            gt_mapped = torch.argmax(all_node_gt, dim=-1)  # (B, A) 0..3

            pred_all.append(pred_mapped.detach().cpu())
            gt_all.append(gt_mapped.detach().cpu())
            prob_all.append(p_ord.detach().cpu())
            observed_all.append(observed_mask.detach().cpu())

            pred_list = pred_mapped.detach().cpu().tolist()  # List[List[int]]，每样本长度6
            gt_list = gt_mapped.detach().cpu().tolist()
            for f, p, g in zip(filenames, pred_list, gt_list):
                csv_rows.append({
                    "filename": f,
                    "pred_label": ",".join(map(str, p)),
                    "gt_label": ",".join(map(str, g)),
                })

    pred_all = torch.cat(pred_all, dim=0).numpy()
    gt_all = torch.cat(gt_all, dim=0).numpy()
    prob_all = torch.cat(prob_all, dim=0).numpy()
    observed_all = torch.cat(observed_all, dim=0).numpy()

    B = len(loader.dataset)
    metric_dicts, count_dict, display_labels = compute_per_attribute_metrics(
        gt_all, pred_all, prob_all, ATTRS, observed_mask=observed_all
    )

    if (epoch + 1) % 10 == 0 and phase in ('val', 'test'):
        out_csv = f"{output_dir}/{dataset}_{phase}_{epoch}.csv"  # 每个 epoch 会覆盖一次；如需按 epoch 存，可自行加上 epoch 编号
        df = pd.DataFrame(csv_rows, columns=["filename", "pred_label", "gt_label"])
        df.to_csv(out_csv, index=False, encoding='utf-8')
        print(f"[Saved] {out_csv} ({len(df)} rows)")

    # ---- 输出信息 ----
    print(f"\n=== {dataset} {phase} finished ===")
    label_width = max(25, max(len(lbl) for lbl in ATTRS))
    print(
        f"{'Label':{label_width}s} {'ImgCount':>10s} {'Accuracy':>10s} {'AUC':>10s} "
        f"{'QWK':>10s} {'Recall':>10s}"
    )
    print("-" * (label_width + 55))
    for lbl in display_labels:
        print(
            f"{lbl:{label_width}s} {count_dict[lbl]:10d} {metric_dicts['acc'][lbl] * 100:9.2f}% "
            f"{metric_dicts['auc'][lbl] * 100:9.2f}% {metric_dicts['qwk'][lbl] * 100:9.2f}% "
            f"{metric_dicts['recall'][lbl] * 100:9.2f}%"
        )
    print("-" * (label_width + 55))
    print(
        f"{'mean':{label_width}s} {'-':>10s} {metric_dicts['acc']['all'] * 100:9.2f}% "
        f"{metric_dicts['auc']['all'] * 100:9.2f}% {metric_dicts['qwk']['all'] * 100:9.2f}% "
        f"{metric_dicts['recall']['all'] * 100:9.2f}%\n"
    )

    return metric_dicts, count_dict, display_labels

    # return acc_dict


if __name__ == '__main__':
    seed = 20200626
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    preprocess2 = _preprocess2()
    preprocess3 = _preprocess3()

    args = parse_args()
    main(args)
