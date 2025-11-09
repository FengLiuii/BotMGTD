

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
import argparse
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns  
from datetime import datetime


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from params import args
from Model_attention import BotMGTD
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from torch.nn.functional import softmax
from DataHandler import DataHandler, index_generator
import pandas as pd  
import pickle
from Utils.Utils import evaluate
import logging  
import random
import dgl
from sklearn.preprocessing import label_binarize  


try:
    from font_config import configure_font
    configure_font()
except ImportError:
   
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 16
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    mpl.rcParams['legend.fontsize'] = 16
    mpl.rcParams['figure.titlesize'] = 16
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    print(" Using default font configuration")

print("DGL backend:", dgl.backend.backend_name)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def set_seed(seed=2025):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AttentionCoach:
   

    def __init__(self, handler):
        self.handler = handler
        self.metrics = dict()
        for met in ['bceLoss', 'AUC']:
            self.metrics['Train' + met] = []
            self.metrics['Test' + met] = []

        
        log_dir = './History/'
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.log_file = os.path.join(log_dir, f"{args.data}_attention_results_{timestamp}.log")

        
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"Attention-based BotMGTD Experiment started at {timestamp}\n")
            f.write(f"Dataset: {args.data}\n")
            f.write(f"Model: Attention-based Relation Fusion\n")
            f.write(f"Hyperparameters:\n")
            for k, v in vars(args).items():
                f.write(f"  {k}: {v}\n")
            
            for k in ['bar_color', 'pie_colors', 'heatmap_cmap', 'bar_width']:
                v = getattr(args, k, None)
                f.write(f"  {k}: {v}\n")
            f.write("\n")

    def run(self):
        
        print(" Starting Attention-based BotMGTD Training")

        for ratio_idx in range(len(self.handler.train_idx)):
            log(f' Ratio Type: {ratio_idx}')
            best_results = {'acc': [], 'macro_f1': [], 'micro_f1': [], 'precision': [], 'recall': [], 'auc': []}

            for repeat in range(1):  
                self.prepareModel()
                log(f' Repeat {repeat+1}/1')

            
                test_lbls = torch.argmax(self.label[self.test_idx[ratio_idx]], dim=-1).detach().cpu().numpy()

                
                best_epoch_results = {'acc': 0, 'macro_f1': 0, 'micro_f1': 0, 'precision': 0, 'recall': 0, 'auc': 0}

                for ep in range(args.epoch):
                    reses = self.trainEpoch(ratio_idx)
                    log(f"🔹 Epoch {ep+1}/{args.epoch} | Train BCE Loss: {reses['bceLoss']:.4f}, Diff Loss: {reses['diffLoss']:.4f}")

                   
                    val_reses, test_reses = self.testEpoch(ratio_idx)

                    
                    test_preds = np.argmax(test_reses['logits'].detach().cpu().numpy(), axis=1)
                    best_proba = softmax(test_reses['logits'], dim=1).detach().cpu().numpy()
                    num_classes = best_proba.shape[1]

                    if num_classes == 2:
                        precision = precision_score(test_lbls, test_preds, average='macro', zero_division=0)
                        recall = recall_score(test_lbls, test_preds, average='macro', zero_division=0)
                        auc = roc_auc_score(y_true=test_lbls, y_score=best_proba[:, 1])
                    else:
                        precision = precision_score(test_lbls, test_preds, average='macro', zero_division=0)
                        recall = recall_score(test_lbls, test_preds, average='macro', zero_division=0)
                        auc = roc_auc_score(y_true=test_lbls, y_score=best_proba, multi_class='ovr')

                    
                    if test_reses['acc'] > best_epoch_results['acc']:
                        best_epoch_results.update({
                            'acc': test_reses['acc'],
                            'macro_f1': test_reses['macro'],
                            'micro_f1': test_reses['micro'],
                            'precision': precision,
                            'recall': recall,
                            'auc': auc
                        })

                    log(f" [Best in Epoch] Acc: {best_epoch_results['acc']:.4f}, Macro-F1: {best_epoch_results['macro_f1']:.4f}, "
                        f"Precision: {best_epoch_results['precision']:.4f}, Recall: {best_epoch_results['recall']:.4f}, AUC: {best_epoch_results['auc']:.4f}")

                
                self.save_attention_analysis(ratio_idx, repeat)

                
                for key in best_results:
                    if isinstance(best_epoch_results[key], torch.Tensor):
                        best_results[key].append(best_epoch_results[key].cpu().numpy())
                    else:
                        best_results[key].append(best_epoch_results[key])

            
            log(f"\n [Final Results for Ratio {ratio_idx}]")
            final_stats = {key: (np.mean(best_results[key]), np.std(best_results[key])) for key in best_results}

            
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"\n [Final Results for Ratio {ratio_idx}]\n")
                f.write(f"🔹 Acc: {final_stats['acc'][0]:.4f} ± {final_stats['acc'][1]:.4f}\n")
                f.write(f"🔹 Macro-F1: {final_stats['macro_f1'][0]:.4f} ± {final_stats['macro_f1'][1]:.4f}\n")
                f.write(f"🔹 Micro-F1: {final_stats['micro_f1'][0]:.4f} ± {final_stats['micro_f1'][1]:.4f}\n")
                f.write(f"🔹 Precision: {final_stats['precision'][0]:.4f} ± {final_stats['precision'][1]:.4f}\n")
                f.write(f"🔹 Recall: {final_stats['recall'][0]:.4f} ± {final_stats['recall'][1]:.4f}\n")
                f.write(f"🔹 AUC: {final_stats['auc'][0]:.4f} ± {final_stats['auc'][1]:.4f}\n")
                f.write("\n")

            log(f" Results saved to {self.log_file}")

    def prepareModel(self):
        
        self.initial_feature = self.handler.feature_list
        self.dim = self.initial_feature.shape[1]
        self.train_idx = self.handler.train_idx
        self.test_idx = self.handler.test_idx
        self.val_idx = self.handler.val_idx
        self.label = self.handler.labels
        self.nbclasses = self.label.shape[1]

        
        if hasattr(self.handler, 'he_adjs'):
            self.he_adjs = self.handler.he_adjs
        else:
            self.he_adjs = [
                getattr(self.handler, attr) for attr in dir(self.handler)
                if attr.startswith("hete_adj") and isinstance(getattr(self.handler, attr), dgl.DGLGraph)
            ]

        if not self.he_adjs:
            raise ValueError("Error: No `hete_adj` found in `DataHandler`. Please check dataset loading.")

        print(f" Found {len(self.he_adjs)} heterogeneous adjacency matrices.")

        
        self.model = BotMGTD(self.dim).to(device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=0)
        
    def trainEpoch(self, ratio_idx=0):
        
        trnLoader = index_generator(batch_size=args.batch, indices=self.train_idx[ratio_idx].cpu().numpy())
        epBCELoss, epDFLoss = 0, 0
        steps = trnLoader.num_iterations()

        for _ in range(steps):
            batch_indices = trnLoader.next()
            ancs = torch.LongTensor(batch_indices).to(device)
            nll_loss, diffloss = self.model.cal_loss(ancs, self.label, self.handler.he_adjs, self.initial_feature)
            epBCELoss += nll_loss.item()
            epDFLoss += diffloss.item()
            self.opt.zero_grad()
            (nll_loss + diffloss).backward()
            self.opt.step()

        return {'bceLoss': epBCELoss / steps, 'diffLoss': epDFLoss / steps}

    def testEpoch(self, ratio_idx):
        
        with torch.no_grad():
            embeds, scores = self.model.get_allembeds(
                self.handler.he_adjs, self.initial_feature,
                label=self.label, save_path=f'{args.data}_attention_emb_label.pt'
            )

            val_acc, val_f1_macro, val_f1_micro, test_acc, test_f1_macro, test_f1_micro, test_logits = evaluate(
                embeds, scores, args.ratio,
                self.train_idx[ratio_idx].cpu().numpy(),
                self.val_idx[ratio_idx].cpu().numpy(),
                self.test_idx[ratio_idx].cpu().numpy(),
                self.label, self.nbclasses
            )

            val_reses = {
                'acc': val_acc, 'macro': val_f1_macro, 'micro': val_f1_micro
            }
            test_reses = {
                'acc': test_acc, 'macro': test_f1_macro, 'micro': test_f1_micro, 'logits': test_logits
            }
            return val_reses, test_reses

    def save_attention_analysis(self, ratio_idx, repeat):
        
        try:
            attention_info = self.model.get_attention_analysis(self.handler.he_adjs, self.initial_feature)
            if attention_info is not None:
                
                analysis_file = f'attention_analysis_ratio_{ratio_idx}_repeat_{repeat}.pkl'
                with open(analysis_file, 'wb') as f:
                    pickle.dump(attention_info, f)

                
                self.create_attention_visualization(attention_info, ratio_idx, repeat)

                log(f" Attention analysis saved to {analysis_file}")
        except Exception as e:
            log(f" Failed to save attention analysis: {e}")

    def create_attention_visualization(self, attention_info, ratio_idx, repeat):
        
        try:
            
            from chart_utils import create_attention_visualization as create_viz
            
            output_file = f'attention_visualization_ratio_{ratio_idx}_repeat_{repeat}.png'
            success = create_viz(attention_info, output_file)
            
            if success:
                log(f" Attention visualization saved: {output_file}")
            else:
                log(f" Failed to create attention visualization: {output_file}")
                
        except ImportError:
            
            log(" Chart utils not available, using fallback method")
            self._create_attention_visualization_fallback(attention_info, ratio_idx, repeat)
        except Exception as e:
            log(f" Failed to create attention visualization: {e}")

    def _create_attention_visualization_fallback(self, attention_info, ratio_idx, repeat):
       
        import numpy as np

        try:
            
            pie_colors = None
            if getattr(args, 'pie_colors', None):
                pie_colors = [c.strip() for c in args.pie_colors.split(',') if c.strip()]

            
            fig = plt.figure(figsize=(18, 12))

            
            first_fusion = attention_info['first_fusion']
            if first_fusion['importance_weights'] is not None:
                first_importance = np.asarray(first_fusion['importance_weights']).mean(axis=0)
                first_names = list(first_fusion['relation_names'])

                
                ax1 = fig.add_subplot(2, 3, 1)
                ax1.bar(first_names, first_importance,
                        color=getattr(args, 'bar_color', None),
                        width=getattr(args, 'bar_width', 0.6))
                ax1.set_title('First Fusion: Friend + Mention + Reply + Quote + Other')
                ax1.set_ylabel('Importance Weight')
                ax1.set_xticklabels(first_names, rotation=45, ha='right')

                
                if first_fusion['attention_weights'] is not None:
                    ax2 = fig.add_subplot(2, 3, 2)
                    data = np.asarray(first_fusion['attention_weights'][0])
                    im = ax2.imshow(data, cmap=getattr(args, 'heatmap_cmap', 'viridis'))
                    ax2.set_title('First Fusion Attention Patterns')
                    ax2.set_xticks(np.arange(len(first_names)))
                    ax2.set_yticks(np.arange(len(first_names)))
                    ax2.set_xticklabels(first_names, rotation=45, ha='right')
                    ax2.set_yticklabels(first_names)
                    
                    mean_val = float(np.mean(data))
                    for i in range(data.shape[0]):
                        for j in range(data.shape[1]):
                            ax2.text(j, i, f"{data[i, j]:.2f}",
                                     ha="center", va="center",
                                     color="white" if data[i, j] > mean_val else "black")
                    cbar = fig.colorbar(im, ax=ax2)
                    cbar.ax.tick_params(labelsize=16)

               
                ax3 = fig.add_subplot(2, 3, 3)
                ax3.pie(first_importance,
                        labels=first_names,
                        autopct='%1.1f%%',
                        colors=pie_colors,
                        textprops={'fontfamily': 'Times New Roman', 'fontsize': 16})
                ax3.set_title('First Fusion Importance Distribution')

            
            second_fusion = attention_info['second_fusion']
            if second_fusion['importance_weights'] is not None:
                second_importance = np.asarray(second_fusion['importance_weights']).mean(axis=0)
                second_names = list(second_fusion['relation_names'])

                
                ax4 = fig.add_subplot(2, 3, 4)
                ax4.bar(second_names, second_importance,
                        color=getattr(args, 'bar_color', None),
                        width=getattr(args, 'bar_width', 0.6))
                ax4.set_title('Second Fusion: Follower + First_Fused')
                ax4.set_ylabel('Importance Weight')
                ax4.set_xticklabels(second_names, rotation=45, ha='right')

                
                if second_fusion['attention_weights'] is not None:
                    ax5 = fig.add_subplot(2, 3, 5)
                    data2 = np.asarray(second_fusion['attention_weights'][0])
                    im2 = ax5.imshow(data2, cmap=getattr(args, 'heatmap_cmap', 'viridis'))
                    ax5.set_title('Second Fusion Attention Patterns')
                    ax5.set_xticks(np.arange(len(second_names)))
                    ax5.set_yticks(np.arange(len(second_names)))
                    ax5.set_xticklabels(second_names, rotation=45, ha='right')
                    ax5.set_yticklabels(second_names)
                    mean_val2 = float(np.mean(data2))
                    for i in range(data2.shape[0]):
                        for j in range(data2.shape[1]):
                            ax5.text(j, i, f"{data2[i, j]:.2f}",
                                     ha="center", va="center",
                                     color="white" if data2[i, j] > mean_val2 else "black")
                    cbar2 = fig.colorbar(im2, ax=ax5)
                    cbar2.ax.tick_params(labelsize=16)

                
                ax6 = fig.add_subplot(2, 3, 6)
                ax6.pie(second_importance,
                        labels=second_names,
                        autopct='%1.1f%%',
                        colors=pie_colors,
                        textprops={'fontfamily': 'Times New Roman', 'fontsize': 16})
                ax6.set_title('Second Fusion Importance Distribution')

            
            fig.tight_layout()
            output_file = f'attention_visualization_ratio_{ratio_idx}_repeat_{repeat}.png'
            fig.savefig(output_file, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            
            log(f" Attention visualization saved: {output_file} (300dpi PNG)")

        except Exception as e:
            log(f" Failed to create attention visualization: {e}")


def build_arg_parser():
    
    parser = argparse.ArgumentParser(description='Run Attention-based BotMGTD Model')
    
    
    parser.add_argument('--data', type=str, default='Twibot20', choices=['Twibot20', 'MGTAB','Twibot22'],
                        help='Dataset to use')
    parser.add_argument('--epoch', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--difflr', type=float, default=1e-2, help='Diffusion learning rate')
    parser.add_argument('--batch', type=int, default=512, help='Batch size')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed')   # 42  3407
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold to filter users')
    
    
    parser.add_argument('--latdim', type=int, default=128, help='Embedding size')
    # parser.add_argument('--uugt_layer', type=int, default=3, help='Number of gt layers')
    parser.add_argument('--uugt_layer', type=int, default=3, help='Number of gt layers')
    parser.add_argument('--head', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--dropRate', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--decay', type=float, default=0.001, help='Weight decay rate')
    
    
    parser.add_argument('--dims', type=str, default='[128]', help='Diffusion model dimensions')
    parser.add_argument('--d_emb_size', type=int, default=8, help='Diffusion embedding size')
    parser.add_argument('--norm', type=bool, default=True, help='Use normalization')
    parser.add_argument('--steps', type=int, default=5, help='Diffusion steps')
    parser.add_argument('--noise_scale', type=float, default=5e-5, help='Noise scale')
    parser.add_argument('--noise_min', type=float, default=0.0001, help='Min noise')
    parser.add_argument('--noise_max', type=float, default=0.001, help='Max noise')
    parser.add_argument('--sampling_steps', type=int, default=0, help='Sampling steps')
    
    
    parser.add_argument('--max_relations', type=int, default=10, help='Maximum number of relations')
    parser.add_argument('--attention_heads', type=int, default=None, help='Number of attention heads (overrides --head)')
    parser.add_argument('--attention_dropout', type=float, default=None, help='Attention dropout (overrides --dropRate)')
    parser.add_argument('--ffn_expansion', type=int, default=4, help='FFN expansion factor')
    parser.add_argument('--importance_hidden_ratio', type=int, default=2, help='Importance predictor hidden ratio')
    parser.add_argument('--weight_init', type=str, default='xavier_uniform', choices=['xavier_uniform', 'xavier_normal'],
                        help='Weight initialization method')
    parser.add_argument('--use_residual', type=bool, default=True, help='Use residual connections')
    parser.add_argument('--denoise_dropout', type=float, default=0.5, help='Denoise model dropout')
    
    
    parser.add_argument('--bar_color', type=str, default=None,
                        help='Bar color, e.g., "#4C72B0" or "tab:blue"')
    parser.add_argument('--pie_colors', type=str, default=None,
                        help='Comma-separated colors for pie slices, e.g., "#4C72B0,#55A868,#C44E52"')
    parser.add_argument('--heatmap_cmap', type=str, default='viridis',
                        help='Matplotlib colormap name for heatmaps')
    parser.add_argument('--bar_width', type=float, default=0.6,
                        help='Bar width for bar charts (0 ~ 1)')

    return parser


def main():
   
    parser = build_arg_parser()
    args_cmd = parser.parse_args()

    
    args.data = args_cmd.data
    args.epoch = args_cmd.epoch
    args.lr = args_cmd.lr
    args.difflr = args_cmd.difflr
    args.batch = args_cmd.batch
    args.patience = args_cmd.patience
    args.threshold = args_cmd.threshold
    
    
    args.latdim = args_cmd.latdim
    args.uugt_layer = args_cmd.uugt_layer
    args.uugt_layer = args_cmd.uugt_layer
    args.head = args_cmd.head
    args.dropRate = args_cmd.dropRate
    args.decay = args_cmd.decay
    
    
    args.dims = args_cmd.dims
    args.d_emb_size = args_cmd.d_emb_size
    args.norm = args_cmd.norm
    args.steps = args_cmd.steps
    args.noise_scale = args_cmd.noise_scale
    args.noise_min = args_cmd.noise_min
    args.noise_max = args_cmd.noise_max
    args.sampling_steps = args_cmd.sampling_steps
    
    
    args.max_relations = args_cmd.max_relations
    args.attention_heads = args_cmd.attention_heads if args_cmd.attention_heads is not None else args_cmd.head
    args.attention_dropout = args_cmd.attention_dropout if args_cmd.attention_dropout is not None else args_cmd.dropRate
    args.ffn_expansion = args_cmd.ffn_expansion
    args.importance_hidden_ratio = args_cmd.importance_hidden_ratio
    args.weight_init = args_cmd.weight_init
    args.use_residual = args_cmd.use_residual
    args.denoise_dropout = args_cmd.denoise_dropout
    
    
    args.bar_color = args_cmd.bar_color
    args.pie_colors = args_cmd.pie_colors
    args.heatmap_cmap = args_cmd.heatmap_cmap
    args.bar_width = args_cmd.bar_width

    
    set_seed(args_cmd.seed)

    print("=" * 60)
    print(" Attention-based BotMGTD Model Training")
    print("=" * 60)
    print(f"Dataset: {args.data}")
    print(f"Epochs: {args.epoch}")
    print(f"Learning Rate: {args.lr}")
    print(f"Batch Size: {args.batch}")
    print(f"Random Seed: {args_cmd.seed}")
    print(f"Bar Color: {args.bar_color}")
    print(f"Pie Colors: {args.pie_colors}")
    print(f"Heatmap Cmap: {args.heatmap_cmap}")
    print(f"Bar Width: {args.bar_width}")
    print("=" * 60)

    
    logger.saveDefault = True
    log(' Starting Attention-based BotMGTD Training')

   
    handler = DataHandler()
    if args.data == "Twibot20":
        handler.load_twibot20_data()
    elif args.data == "MGTAB":
        handler.load_mgtab_data()
    elif args.data == "twibot22":
        handler.load_twibot20_data()
    else:
        raise ValueError(f"Unsupported dataset: {args.data}")

    log(f' Loaded {args.data} Data')

    
    coach = AttentionCoach(handler)

    
    try:
        coach.run()
        print("\n Training completed successfully!")
        print(f" Results saved to: {coach.log_file}")
        print(" Attention visualizations saved as PNG files")

    except Exception as e:
        print(f" Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
