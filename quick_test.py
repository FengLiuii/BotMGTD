#!/usr/bin/env python3


import os
import sys
import torch
import numpy as np


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from params import args
from Model_attention import BotMGTD
from DataHandler import DataHandler
import dgl

def quick_test():
   
    print("🧪 Quick Test for Attention-based BotMGTD Model")
    print("=" * 50)
    
   
    args.data = 'Twibot20'
    args.epoch = 5  
    args.lr = 1e-2
    args.batch = 256
    
    print(f"Dataset: {args.data}")
    print(f"Test Epochs: {args.epoch}")
    print(f"Batch Size: {args.batch}")
    
    try:
        
        print("\n📁 Loading data...")
        handler = DataHandler()
        handler.load_twibot20_data()
        print(f"✅ Data loaded successfully")
        print(f"   - Features shape: {handler.feature_list.shape}")
        print(f"   - Number of relations: {len(handler.he_adjs)}")
        print(f"   - Train samples: {len(handler.train_idx[0])}")
        print(f"   - Test samples: {len(handler.test_idx[0])}")
        
        
        print("\n🏗️ Creating model...")
        model = BotMGTD(handler.feature_list.shape[1]).to('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✅ Model created successfully")
        print(f"   - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        
        print("\n🔄 Testing forward pass...")
        with torch.no_grad():
            forward_result = model.forward(handler.he_adjs, handler.feature_list)
            if isinstance(forward_result, tuple) and len(forward_result) == 4:
                aux_fused, target_emb, attention_weights, importance_weights = forward_result
            else:
                aux_fused, target_emb = forward_result
                attention_weights, importance_weights = None, None
                
            print(f"✅ Forward pass successful")
            print(f"   - Target embedding shape: {target_emb.shape}")
            print(f"   - Auxiliary fused shape: {aux_fused.shape}")
            if attention_weights is not None:
                if isinstance(attention_weights, tuple):
                    print(f"   - Attention weights (tuple): {len(attention_weights)} elements")
                    for i, attn in enumerate(attention_weights):
                        if hasattr(attn, 'shape'):
                            print(f"     - Attention {i} shape: {attn.shape}")
                else:
                    print(f"   - Attention weights shape: {attention_weights.shape}")
            if importance_weights is not None:
                if isinstance(importance_weights, tuple):
                    print(f"   - Importance weights (tuple): {len(importance_weights)} elements")
                    for i, imp in enumerate(importance_weights):
                        if hasattr(imp, 'shape'):
                            print(f"     - Importance {i} shape: {imp.shape}")
                else:
                    print(f"   - Importance weights shape: {importance_weights.shape}")
        
        
        print("\n🏃 Testing training step...")
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        
        train_indices = handler.train_idx[0][:args.batch].cpu().numpy()
        batch_indices = torch.LongTensor(train_indices).to('cuda' if torch.cuda.is_available() else 'cpu')
        
        
        nll_loss, diff_loss = model.cal_loss(batch_indices, handler.labels, handler.he_adjs, handler.feature_list)
        total_loss = nll_loss + diff_loss
        
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        print(f"✅ Training step successful")
        print(f"   - NLL Loss: {nll_loss.item():.4f}")
        print(f"   - Diffusion Loss: {diff_loss.item():.4f}")
        print(f"   - Total Loss: {total_loss.item():.4f}")
        
        
        print("\n🔍 Testing attention analysis...")
        attention_info = model.get_attention_analysis(handler.he_adjs, handler.feature_list)
        if attention_info is not None:
            print(f"✅ Attention analysis successful")
            #print(f"   - Relation importance shape: {attention_info['relation_importance']}")
            #print(f"   - Mean importance weights: {attention_info['relation_importance'].mean(axis=0)}")
        else:
            print("⚠️ No attention information available")
        
        print("\n🎉 All tests passed! Model is ready for training.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == '__main__':
    success = quick_test()
    if success:
        print("\n✅ Quick test completed successfully!")
        print("You can now run the full training with: python run_attention_model.py")
    else:
        print("\n❌ Quick test failed. Please check the error messages above.")
