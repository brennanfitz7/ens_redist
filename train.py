import torch 
import glob
import sys
import argparse
import os
import pandas as pd
import numpy as np
import torch.nn as nn

from data_loader import EnsDataLoader

from model import both_structs


def accuracy(pred_y, y, w=0.2):
    """Calculate accuracy."""
    c=0
    for i in range(0,len(y)):
        if y[i]-0.2 < pred_y[i] < y[i]+0.2:
            c+=1
    return c/ len(y)

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, target):
        criterion = nn.MSELoss()
        loss = criterion(output, target)
        mask = target >=2
        corr_loss=1-(torch.corrcoef(torch.stack([output,target]))[0][1])
        high_cost = (loss * mask.float()).mean() 
        
        return loss + (10**6)*high_cost + (10**6)*corr_loss

def train(train_dir, val_dir, outdir, use_stored_seed=True):
    """Train a GatConv model and return the trained model."""

    os.mkdir(outdir)
    
    if use_stored_seed==True:
        seed = 81
        torch.manual_seed(seed)

    model=both_structs(4)

    train_data=EnsDataLoader(glob.glob(train_dir+'/*/'),batch_size=100)
    val_data=EnsDataLoader(glob.glob(val_dir+'/*/'),batch_size=100)
    
    criterion = CustomLoss()
    optimizer = model.optimizer
    epochs = 200

    train_acc_list=[]
    train_loss_list=[]
    val_loss_list=[]
    val_acc_list=[]
    model.train()
    for epoch in range(epochs+1):
        # Training
        optimizer.zero_grad()
        out = model(train_data.dataset1,train_data.dataset2)
        loss = criterion(out.squeeze(), train_data.labels)
        acc = accuracy(out.squeeze(), train_data.labels)
        loss.backward()
        optimizer.step()

        # Validation
        val_out=model(val_data.dataset1, val_data.dataset2)
        val_loss = criterion(val_out.squeeze(), val_data.labels)
        val_acc = accuracy(val_out.squeeze(), val_data.labels)

        train_acc_list.append(acc)
        train_loss_list.append(loss.item())
        val_loss_list.append(val_loss.item())
        val_acc_list.append(val_acc)

        # Print metrics every 10 epochs
        if(epoch % 10 == 0):
            print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: '
                  f'{acc*100:>6.2f}% | Val Loss: {val_loss:.2f} | '
                  f'Val Acc: {val_acc*100:.2f}%')

    stats_tracking=pd.DataFrame({'train_loss':train_loss_list,
                                 'train_acc':train_acc_list,
                                 'val_loss':val_loss_list,
                                 'val_acc':val_acc_list})
    
    train_output=pd.DataFrame({'real_train':train_data.labels.tolist(),
                              'pred_train':out.squeeze().tolist()})
    val_output=pd.DataFrame({'real_val':val_data.labels.tolist(),
                              'pred_val':val_out.squeeze().tolist()})

    stats_tracking.to_csv(outdir+'/stats_tracking.csv',index=False)

    train_output.to_csv(outdir+'/train_output.csv',index=False)
    val_output.to_csv(outdir+'/val_output.csv',index=False)

    torch.save(model.state_dict(), outdir+'/trained_model.pt')

    return model, train_output, val_output, stats_tracking


def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument("--train_data","-t",required=True,
                        help="training data directory")
    parser.add_argument("--val_data","-v",required=True,
                        help="validation data directory")
    parser.add_argument("--outdir","-o",required=True,
                        help="outdir where your outputs will go")
    parser.add_argument("--verbose", action="store_true",
                    help="increase output verbosity")
    parser.add_argument("--use_stored_seed", action="store_true", required=False,
                        help="use the seed stored in this file. Will reproduce previously trained model. Default: True")
    args = parser.parse_args(argv)
    
    train(train_dir=args.train_data, 
          val_dir=args.val_data,
          outdir=args.outdir,
          use_stored_seed=args.use_stored_seed)


if __name__ == "__main__":
    main()


