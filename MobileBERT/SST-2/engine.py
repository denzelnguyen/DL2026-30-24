import torch
from sklearn.metrics import accuracy_score, f1_score
import config

def train_one_epoch(student, teacher, dataloader, optimizer, scheduler, criterion):
    student.train()
    running_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(config.DEVICE)
        attention_mask = batch["attention_mask"].to(config.DEVICE)
        token_type_ids = batch.get("token_type_ids", None)
        if token_type_ids is not None: token_type_ids = token_type_ids.to(config.DEVICE)
        labels = batch["labels"].to(config.DEVICE)

        with torch.no_grad():
            teacher_outputs = teacher(input_ids=input_ids, attention_mask=attention_mask)
            teacher_probs = torch.softmax(teacher_outputs.logits, dim=-1)[:, 1]

        student_logits = student(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        loss_hard = criterion(student_logits, labels)
        loss_soft = criterion(student_logits, teacher_probs)
        loss = (1.0 - config.ALPHA) * loss_hard + config.ALPHA * loss_soft

        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        running_loss += loss.item() * labels.size(0)
    return running_loss

def evaluate(student, dataloader, criterion):
    student.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(config.DEVICE)
            attention_mask = batch["attention_mask"].to(config.DEVICE)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None: token_type_ids = token_type_ids.to(config.DEVICE)
            labels = batch["labels"].to(config.DEVICE)

            student_logits = student(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = criterion(student_logits, labels)
            running_loss += loss.item() * labels.size(0)

            probs = torch.sigmoid(student_logits)
            all_preds.extend((probs >= 0.5).long().cpu().numpy())
            all_labels.extend(labels.long().cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return running_loss, acc, f1