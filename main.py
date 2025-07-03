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
import os
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def setup_logging():

    log_filename = f"log/app_{current_time}.log"

    logging.basicConfig(level=logging.INFO,
                        handlers=[
                            logging.FileHandler(log_filename),
                            logging.StreamHandler()
                        ])
def log_selected_args(args):
    """记录选定的参数"""

    interested_args = ['batch_size','lr','l2','num_layers','model']
    for arg in interested_args:
        if hasattr(args, arg):
            logging.info(f"参数 {arg}: {getattr(args, arg)}")

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
def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
def plot_prediction(y_true, y_pred, out_put_len, model,idx=100):
    plt.figure(figsize=(10, 4))

    y_true_norm = normalize(y_true[idx])
    y_pred_norm = normalize(y_pred[idx])

    plt.plot(y_true_norm, label='True')
    plt.plot(y_pred_norm, label='Predicted')
    plt.legend()
    plt.title(f"Sample_{model}_{out_put_len}_Prediction")
    plt.savefig(f"Sample_{model}_{out_put_len}_Prediction.pdf", bbox_inches='tight', dpi=300, format='pdf')

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
    log_selected_args(init_args)

    file_path = 'dataset/train_cleaned.csv'
    data = load_and_process(file_path)
    file_path = 'dataset/test_cleaned.csv'
    test_data = load_and_process(file_path)
    if not init_args.test:
        if init_args.model=='LSTM' or init_args.model =='Transformer':
            inputs, labels = create_sequences(data, input_len=init_args.input_len, output_len=init_args.output_len)
            dataset = Dataset(inputs, labels)
            train_loader, val_loader = split_train_val(dataset.X, dataset.Y, batch_size=init_args.batch_size,
                                                       shuffle=True)
        else:
            all_values = get_all_values(data, test_data)
            encoder = PositionalEncoder(output_size=64, all_values=all_values, n=10000, scaler=init_args.scalar)
            inputs, labels = create_logic_sequences(data, input_len=init_args.input_len, output_len=init_args.output_len)
            powers = create_power_sequences(data, input_len=init_args.input_len, output_len=init_args.output_len)
            dataset = LogicalDataset(inputs, labels, powers)
            train_loader, val_loader = split_train_val_logic(dataset.X, dataset.Y, dataset.P, batch_size=init_args.batch_size,
                                                       shuffle=True)
        if init_args.model=='LSTM':
            model = LSTM(input_size=inputs.shape[2], hidden_size=64, num_layers=init_args.num_layers, output_len=init_args.output_len)
            train_model_with_val(model,train_loader,val_loader,epochs=init_args.epoch,model_path=init_args.saved_path,model_name=init_args.model, lr=init_args.lr, l2=init_args.l2, output_len=init_args.output_len)
        elif init_args.model=='Transformer':
            model = Transformer(input_size=inputs.shape[2], d_model=64, nhead=4, num_layers=init_args.num_layers, output_len=init_args.output_len)
            train_model_with_val(model,train_loader,val_loader,epochs=init_args.epoch,model_path=init_args.saved_path,model_name=init_args.model, lr=init_args.lr, l2=init_args.l2, output_len=init_args.output_len)
        else:
            model = LogicalTransformer(input_size=inputs.shape[2], d_model=64, nhead=4, num_layers=init_args.num_layers,
                                         output_len=init_args.output_len, encoder=encoder)
            train_logic_model_with_val(model,train_loader,val_loader,epochs=init_args.epoch,model_path=init_args.saved_path,model_name=init_args.model, lr=init_args.lr, l2=init_args.l2, output_len=init_args.output_len)

    # test
    if init_args.model == 'LSTM' or init_args.model == 'Transformer':
        test_inputs, test_labels = create_sequences(test_data, input_len=init_args.input_len, output_len=init_args.output_len)
        test_dataset = Dataset(test_inputs, test_labels)
        test_dataloader = DataLoader(test_dataset, batch_size=init_args.batch_size, shuffle=False)
    else:
        test_inputs, test_labels = create_logic_sequences(test_data, input_len=init_args.input_len, output_len=init_args.output_len)
        test_powers = create_power_sequences(test_data, input_len=init_args.input_len, output_len=init_args.output_len)
        test_dataset = LogicalDataset(test_inputs, test_labels, test_powers)
        test_dataloader = DataLoader(test_dataset, batch_size=init_args.batch_size, shuffle=False)

    if init_args.model == 'LSTM':
        model = LSTM(input_size=test_inputs.shape[2], hidden_size=64, num_layers=init_args.num_layers, output_len=init_args.output_len)
    elif init_args.model == 'Transformer':
        model = Transformer(input_size=test_inputs.shape[2], d_model=64, nhead=4, num_layers=init_args.num_layers,
                            output_len=init_args.output_len)
    else:
        all_values = get_all_values(data, test_data)
        encoder = PositionalEncoder(output_size=64, all_values=all_values, n=10000, scaler=init_args.scalar)
        model = LogicalTransformer(input_size=test_inputs.shape[2], d_model=64, nhead=4, num_layers=init_args.num_layers,
                                     output_len=init_args.output_len, encoder=encoder)
    model_path = f"saved/best_{init_args.model}_{init_args.output_len}.json"
    model.load_state_dict(torch.load(model_path))
    if init_args.model=='LSTM' or init_args.model =='Transformer':
        mse, mae, y_true, y_pred = evaluate_model(model, test_dataloader)
        logging.info(f"MSE:{mse},MAE:{mae}")
    else:
        mse, mae, y_true, y_pred = evaluate_logic_model(model, test_dataloader)
        logging.info(f"MSE:{mse},MAE:{mae}")

    # visualization

    # plot_prediction(y_true, y_pred, init_args.output_len, init_args.model)

if __name__ == '__main__':
    setup_logging()
    main()