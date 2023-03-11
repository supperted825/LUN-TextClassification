# Layer-wise Learning Rate Decay (LLRD) for Transformer Finetuning
# Code adapted from https://towardsdatascience.com/advanced-techniques-for-fine-tuning-transformers-82e4e61e16e

import torch


def LayerwiseLR(model, init_lr, decay_factor):
    
    opt_parameters = []    # To be passed to the optimizer (only parameters of the layers you want to update).
    named_parameters = list(model.named_parameters())
        
    # According to AAAMLP book by A. Thakur, we generally do not use any decay for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    lr = head_lr = init_lr
    
    # === Pooler and regressor ======================================================  
    
    params_0 = [p for n, p in named_parameters if ("pooler" in n or "regressor" in n) 
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n, p in named_parameters if ("pooler" in n or "regressor" in n)
                and not any(nd in n for nd in no_decay)]
    
    head_params = {"params": params_0, "lr": head_lr, "weight_decay": 0.0}    
    opt_parameters.append(head_params)
        
    head_params = {"params": params_1, "lr": head_lr, "weight_decay": 0.01}    
    opt_parameters.append(head_params)
                
    # === 12 Hidden layers ==========================================================
    
    for layer in range(11, -1, -1):
        
        params_0 = [p for n, p in named_parameters if f"encoder.layer.{layer}." in n 
                    and any(nd in n for nd in no_decay)]
        params_1 = [p for n, p in named_parameters if f"encoder.layer.{layer}." in n 
                    and not any(nd in n for nd in no_decay)]
        
        layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
        opt_parameters.append(layer_params)   
                            
        layer_params = {"params": params_1, "lr": lr, "weight_decay": 0.01}
        opt_parameters.append(layer_params)       
        
        lr *= decay_factor    
        
    # === Embeddings layer ==========================================================
    
    params_0 = [p for n, p in named_parameters if "embeddings" in n 
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n, p in named_parameters if "embeddings" in n
                and not any(nd in n for nd in no_decay)]
    
    embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0} 
    opt_parameters.append(embed_params)
        
    embed_params = {"params": params_1, "lr": lr, "weight_decay": 0.01} 
    opt_parameters.append(embed_params)        
    
    return torch.optim.AdamW(opt_parameters, lr=init_lr, eps=1e-8)


def LowerBackboneLR(model, init_lr):
    return torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": 0.01 * init_lr},
            {"params": model.fc1.parameters()},
            {"params": model.fc2.parameters()},
            {"params": model.fc3.parameters()},
        ],
        lr = init_lr
    )