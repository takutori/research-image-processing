from __future__ import annotations

from collections.abc import Callable

import torch
from tqdm.auto import tqdm


def _extract_logits(model_output):
    return model_output.logits if hasattr(model_output, "logits") else model_output


def _prepare_model_inputs(batch: dict[str, object], device: str):
    if "model_inputs" in batch:
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch["model_inputs"].items()
        }
    return batch["image"].to(device)


def _forward_model(model, batch: dict[str, object], device: str):
    model_inputs = _prepare_model_inputs(batch, device)
    if isinstance(model_inputs, dict):
        return model(**model_inputs)
    return model(model_inputs)


@torch.no_grad()
def evaluate_one_epoch(model, dataloader, criterion, device: str = "cuda") -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in tqdm(dataloader, desc="eval", leave=False):
        labels = batch["label"].to(device)

        logits = _extract_logits(_forward_model(model, batch, device))
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += labels.size(0)

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
    }


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device: str = "cuda",
    scheduler=None,
    scaler: torch.amp.GradScaler | None = None,
    log_interval: int = 20,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    progress = tqdm(dataloader, desc="train", leave=False)
    for step, batch in enumerate(progress, start=1):
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                logits = _extract_logits(_forward_model(model, batch, device))
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = _extract_logits(_forward_model(model, batch, device))
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

        if step % log_interval == 0 or step == len(dataloader):
            progress.set_postfix(
                loss=f"{total_loss / max(total_examples, 1):.4f}",
                acc=f"{total_correct / max(total_examples, 1):.4f}",
            )

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
    }


def fit_classifier(
    model,
    train_dataloader,
    valid_dataloader,
    criterion,
    optimizer,
    num_epochs: int,
    device: str = "cuda",
    scheduler=None,
    scaler: torch.amp.GradScaler | None = None,
    epoch_end_callback: Callable[[int, dict[str, float]], None] | None = None,
) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            scaler=scaler,
        )
        valid_metrics = evaluate_one_epoch(
            model=model,
            dataloader=valid_dataloader,
            criterion=criterion,
            device=device,
        )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "valid_loss": valid_metrics["loss"],
            "valid_accuracy": valid_metrics["accuracy"],
        }
        history.append(epoch_metrics)

        if epoch_end_callback is not None:
            epoch_end_callback(epoch, epoch_metrics)

    return history
