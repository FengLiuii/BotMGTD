
```
├── Model_attention.py          # attention based model
├── run_attention_model.py      # train
├── quick_test.py              # quick test
├── run_attention.sh           # Linux/Mac shell
├── README_attention.md        
└── aggregation_comparison.png 
```



**Linux/Mac:**
```bash
./run_attention.sh
```



**2. train**
```bash
python run_attention_model.py --data Twibot20 --epoch 200 --lr 0.001 --batch 256
```


```bash
python run_attention_model.py \
    --data MGTAB \
    --epoch 200 \
    --lr 0.005 \
    --batch 256 \
    --seed 123
```

## para state

| para | value | instand |
|------|--------|------|
| `--data` | Twibot20 | datasets (Twibot20/MGTAB/Twibot22) |
| `--epoch` | 100 | train epoch |
| `--lr` | 0.01 | learn rate |
| `--batch` | 512 | batch |
| `--seed` | 42 | random seed |


## dataset
```data/
├── Twibot-20/
│   ├── a_feature.npz   # Account features; processing follows BotRGCN
│   │                   # https://github.com/LuoUndergradXJTU/TwiBot-22/tree/master/src/BotRGCN/twibot_20
│   ├── follower.npz    # Follower graph
│   ├── friend.npz      # Friend graph
│   └── label.npz       # Account labels
│
├── MGTAB/
│   ├── a_feature.npz   # Account features; processing follows MGTAB
│   │                   # https://github.com/GraphDetec/MGTAB/blob/main/README.md
│   ├── follower.npz    # Follower graph
│   ├── friend.npz      # Friend graph
│   ├── label.npz       # Account labels
│   ├── mention.npz     # Mention graph
│   ├── quote.npz       # Quote graph
│   ├── reply.npz       # Reply graph
│   └── other.npz       # Other interaction graph
│
└── Twibot-22/
    ├── a_feature.npz   # Account features; processing follows BotRGCN (TwiBot-22)
    │                   # https://github.com/LuoUndergradXJTU/TwiBot-22/tree/master/src/BotRGCN/twibot_22
    ├── follower.npz    # Follower graph
    ├── friend.npz      # Friend graph
    └── label.npz       # Account labels
```


## output file

### 1. log
- `History/{dataset}_attention_results_{timestamp}.log`


### 2. attention analysis
- `attention_analysis_ratio_{ratio}_repeat_{repeat}.pkl`


### 3. attention vis
- `attention_visualization_ratio_{ratio}_repeat_{repeat}.png`


### 4. embedding and label
- `{dataset}_attention_emb_label.pt`


## model

### attention model
```python
class AttentionBasedRelationFusion(nn.Module):
    def __init__(self, hidden_dim, num_relations, num_heads=8):
        
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim)
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)
        
        self.importance_predictor = nn.Linear(hidden_dim, 1)
```

### Aggregation Process
1. **Relation Encoding**: Each relational graph is encoded independently.  
2. **Attention Aggregation**: All relation types are processed in parallel.  
3. **Importance Learning**: Relation importance weights are automatically learned.  
4. **Final Fusion**: Target and auxiliary embeddings are fused through concatenation and an MLP.  

## Experimental Results

### Interpretability Analysis
- **Relation Importance**: Automatically learn the importance weights of different relations.  
- **Attention Patterns**: Visualize the interaction patterns among relations.  
- **Weight Distribution**: Analyze the distribution characteristics of attention weights.  

## Citation

If you use this attention aggregation version, please cite:


