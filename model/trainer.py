import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging

def train_model(model, train_loader, epochs=20, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        losses = []
        for X_batch, Y_batch in train_loader:
            pred = model(X_batch)
            loss = criterion(pred, Y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1}: Loss = {np.mean(losses):.4f}")


def train_model_with_val(model, train_loader, val_loader, epochs, model_path, model_name, lr, l2, output_len):
    model_path=f"{model_path}best_{model_name}_{output_len}.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    best_val_loss = float('inf')
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # 验证阶段
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())
        val_loss = sum(val_losses) / len(val_losses)

        logging.info(
            f"[Epoch {epoch + 1}/{epochs}] Train Loss: {sum(train_losses) / len(train_losses):.4f} | Val Loss: {val_loss:.4f}")

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)

    logging.info(f"Best model saved at epoch {best_epoch + 1} with val loss = {best_val_loss:.4f}")

def generate_negative_labels_noise(labels, noise_scale=0.2):
    """
    对每个真实值添加扰动（高斯或均匀）作为负样本
    """
    noise = torch.randn_like(labels) * noise_scale  # 正态扰动
    labels_neg = labels + noise
    return labels_neg
def train_logic_model_with_val(model, train_loader, val_loader, epochs, model_path, model_name, lr, l2, output_len):
    model_path=f"{model_path}best_{model_name}_{output_len}.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    best_val_loss = float('inf')
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb, pb in train_loader:
            xb, yb, pb = xb.to(device), yb.to(device), pb.to(device)
            pred = model(xb, pb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # 验证阶段
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb, pb in val_loader:
                xb, yb, pb = xb.to(device), yb.to(device), pb.to(device)
                pred = model(xb, pb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())
        val_loss = sum(val_losses) / len(val_losses)

        logging.info(
            f"[Epoch {epoch + 1}/{epochs}] Train Loss: {sum(train_losses) / len(train_losses):.4f} | Val Loss: {val_loss:.4f}")

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)

    logging.info(f"Best model saved at epoch {best_epoch + 1} with val loss = {best_val_loss:.4f}")
