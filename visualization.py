from torch.utils.data import DataLoader
from data.dataprocess import load_and_process,create_sequences,Dataset,split_train_val,create_logic_sequences,create_power_sequences,LogicalDataset,split_train_val_logic,get_all_values
from model.LSTM import LSTM
from model.Transformer import Transformer
from model.LogicalTransformer import LogicalTransformer
from model.trainer import train_model_with_val, train_logic_model_with_val
import argparse
import logging
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime
from model.numeral_encoder import PositionalEncoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            pred = model(X_batch)
            all_preds.append(pred.numpy()) #[bs,output_len]
            all_targets.append(Y_batch.numpy())
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    return mse, mae, targets, preds

def evaluate_logic_model(model, test_loader):
    model.eval()
    model.to(device)
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, Y_batch, P_batch in test_loader:
            X_batch, Y_batch, P_batch = X_batch.to(device), Y_batch.to(device), P_batch.to(device)
            pred = model(X_batch, P_batch)
            all_preds.append(pred.detach().cpu().numpy())
            all_targets.append(Y_batch.detach().cpu().numpy())
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    return mse, mae, targets, preds
def compare_models(model_dict, test_loader1, test_loader2, device='cuda'):
    """
    model_dict: dict[str, nn.Module], e.g. {'LSTM': model1, 'Transformer': model2, ...}
    """
    results = {}

    for name, model in model_dict.items():
        print(f"Evaluating {name}...")
        if name == 'LSTM' or name == 'Transformer':
            mse, mae, _ , _ = evaluate_model(model, test_loader1)
        else:
            mse, mae, _ , _ = evaluate_logic_model(model, test_loader2)
        results[name] = {'mse': mse, 'mae': mae}
        print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")

    return results

def plot_metric_separate(results, output_len):
    models = list(results.keys())
    mses = [results[m]['mse'] for m in models]
    maes = [results[m]['mae'] for m in models]

    x = np.arange(len(models))

    plt.figure(figsize=(8, 4))
    plt.bar(x, mses, width=0.3)
    plt.xticks(x, models)
    plt.ylabel("MSE")
    plt.title("Model MSE Comparison")
    plt.tight_layout()
    plt.savefig(f"model_mse_comparison_{output_len}.pdf", dpi=300)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.bar(x, maes, width=0.3, color='orange')
    plt.xticks(x, models)
    plt.ylabel("MAE")
    plt.title("Model MAE Comparison")
    plt.tight_layout()
    plt.savefig(f"model_mae_comparison_{output_len}.pdf", dpi=300)
    plt.show()
def main():
    # init args
    init_parser = argparse.ArgumentParser(description='Experiment')
    init_parser.add_argument('--epoch', type=int, default=100,
                             help='Number of epochs for the training of the model')
    init_parser.add_argument('--batch_size', type=int, default=32,
                             help='Size of batch for the training of the model')
    init_parser.add_argument('--lr', type=float, default=0.001,
                             help='Learning rate for the training of the model.')
    init_parser.add_argument('--l2', type=float, default=0.00001,
                             help='Weight decay for the training of the model.')
    init_parser.add_argument('--emb_size', type=int, default=12,
                             help='Size of users, item, and event embeddings.')
    init_parser.add_argument('--saved_path', type=str, default="saved/",
                             help='Path where the model has to be saved during training. '
                                  'The model is saved every time the validation metric increases. '
                                  'This path is also used for loading the best model before the test evaluation.')
    init_parser.add_argument('--test', type=bool, default=True,
                             help='Whether training the model.')
    init_parser.add_argument('--model', type=str, default="Logical",
                             help='(LSTM,Transformer,Logical).')
    init_parser.add_argument('--num_layers', type=int, default=10,
                             help='Layers of the model.')
    init_parser.add_argument('--input_len', type=int, default="90",
                             help='Length of the historical inputs.')
    init_parser.add_argument('--output_len', type=int, default="90",
                             help='Length of the prediction.')

    init_parser.add_argument('--tau', type=float, default=0.05,
                             help='Length of the prediction.')
    init_parser.add_argument('--gamma', type=int, default=1,
                             help='Length of the prediction.')
    init_parser.add_argument('--scalar', type=str, default='log',
                             help='Scalar type')
    init_args, init_extras = init_parser.parse_known_args()

    file_path = 'dataset/train_cleaned.csv'
    data = load_and_process(file_path)
    file_path = 'dataset/test_cleaned.csv'
    test_data = load_and_process(file_path)

    test_inputs, test_labels = create_sequences(test_data, input_len=init_args.input_len, output_len=init_args.output_len)
    test_dataset = Dataset(test_inputs, test_labels)
    test_dataloader1 = DataLoader(test_dataset, batch_size=init_args.batch_size, shuffle=False)
    model1 = LSTM(input_size=test_inputs.shape[2], hidden_size=64, num_layers=init_args.num_layers,
                  output_len=init_args.output_len)
    model_path1 = f"saved/best_LSTM_90.json"
    model1.load_state_dict(torch.load(model_path1))
    model2 = Transformer(input_size=test_inputs.shape[2], d_model=64, nhead=4, num_layers=init_args.num_layers,
                         output_len=init_args.output_len)
    model_path2 = f"saved/best_Transformer_90.json"
    model2.load_state_dict(torch.load(model_path2))

    all_values = get_all_values(data, test_data)
    encoder = PositionalEncoder(output_size=64, all_values=all_values, n=10000, scaler=init_args.scalar)
    test_inputs, test_labels = create_logic_sequences(test_data, input_len=init_args.input_len, output_len=init_args.output_len)
    test_powers = create_power_sequences(test_data, input_len=init_args.input_len, output_len=init_args.output_len)
    test_dataset = LogicalDataset(test_inputs, test_labels, test_powers)
    test_dataloader2 = DataLoader(test_dataset, batch_size=init_args.batch_size, shuffle=False)

    model3 = LogicalTransformer(input_size=test_inputs.shape[2], d_model=64, nhead=4, num_layers=init_args.num_layers,
                                output_len=init_args.output_len, encoder=encoder)
    model_path3 = f"saved/best_Logical_90.json"
    model3.load_state_dict(torch.load(model_path3))

    models = {
        'LSTM': model1,
        'Transformer': model2,
        'Logical': model3
    }
    results = compare_models(models, test_dataloader1, test_dataloader2, device='cuda')
    # 可视化结果
    plot_metric_separate(results, init_args.output_len)

if __name__ == '__main__':
    main()