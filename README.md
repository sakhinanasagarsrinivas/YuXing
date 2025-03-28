# # #####################################################################
# # #############             Testing
# # #####################################################################    

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers import BitsAndBytesConfig

# # Configure 4-bit quantization
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,  # or torch.bfloat16 if your GPU supports it
#     bnb_4bit_quant_type="nf4",  # Can be "nf4" or "fp4"
#     bnb_4bit_use_double_quant=True
# )

# # Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# # Load model with quantization config
# model = AutoModelForCausalLM.from_pretrained(
#     "meta-llama/Llama-3.2-1B-Instruct",
#     quantization_config=quantization_config,
#     device_map="auto"
# )

# # Print the model to see its structure
# print(model)

# # Get the model configuration
# config = model.config

# # Print the configuration in the desired format
# print("DEFAULT_CONFIG = {")
# print(f"    \"vocab_size\": {config.vocab_size},      # Vocabulary size")
# print(f"    \"context_length\": {config.max_position_embeddings},  # Context length")
# print(f"    \"emb_dim\": {config.hidden_size},            # Embedding dimension")
# print(f"    \"n_heads\": {config.num_attention_heads},              # Number of attention heads")
# print(f"    \"n_layers\": {config.num_hidden_layers},             # Number of layers")
# print(f"    \"hidden_dim\": {config.intermediate_size},         # Size of the intermediate dimension in FeedForward")
# print(f"    \"n_kv_groups\": {config.num_key_value_heads},           # Key-Value groups for grouped-query attention")

# # RoPE parameters (some might need to be extracted differently depending on model version)
# rope_base = getattr(config, "rope_theta", 500000.0)
# print(f"    \"rope_base\": {rope_base},     # The base in RoPE's \"theta\"")
# print(f"    \"dtype\": torch.bfloat16,    # Lower-precision dtype to reduce memory usage")

# # Print RoPE frequency information if available
# print("    \"rope_freq\": {              # RoPE frequency scaling")
# print("        \"factor\": 32.0,")
# print("        \"low_freq_factor\": 1.0,")
# print("        \"high_freq_factor\": 4.0,")
# print("        \"original_context_length\": 8192,")
# print("    }")
# print("}")


#####################################################################
#############              Architecture code
#####################################################################    

import os
import math
import torch
import tiktoken
import torch.nn as nn
from pathlib import Path
from safetensors.torch import load_file
from tiktoken.load import load_tiktoken_bpe
from huggingface_hub import login, hf_hub_download

class Tokenizer:
    def __init__(self, model_path):
        assert os.path.isfile(model_path), f"Model file {model_path} not found"
        mergeable_ranks = load_tiktoken_bpe(model_path)

        # Special tokens with fixed IDs
        self.special_tokens = {
            "<|begin_of_text|>": 128000,
            "<|end_of_text|>": 128001,
            "<|start_header_id|>": 128006,
            "<|end_header_id|>": 128007,
            "<|eot_id|>": 128009,
        }
        
        # Add reserved tokens
        self.special_tokens.update({
            f"<|reserved_{i}|>": 128002 + i 
            for i in range(256) 
            if (128002 + i) not in self.special_tokens.values()
        })

        # Initialize tiktoken encoding
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens
        )

    def encode(self, text, bos=False, eos=False, allowed_special=set(), disallowed_special=()):
        """Encode text to token IDs."""
        tokens = []
        if bos:
            tokens.append(self.special_tokens["<|begin_of_text|>"])
        
        tokens.extend(self.model.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special))
        
        if eos:
            tokens.append(self.special_tokens["<|end_of_text|>"])
        return tokens

    def decode(self, tokens):
        """Decode token IDs back to text."""
        return self.model.decode(tokens)

class ChatFormat:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def encode_header(self, message):
        """Encode message header with role information."""
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
        tokens.extend(self.tokenizer.encode(message["role"], bos=False, eos=False))
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        return tokens
        
    def encode(self, text):
        """Encode a complete message with role and content."""
        message = {
            "role": "user",
            "content": text
        }
        
        tokens = self.encode_header(message)
        tokens.extend(self.tokenizer.encode(message["content"].strip(), bos=False, eos=False))
        tokens.append(self.tokenizer.special_tokens["<|eot_id|>"])
        return tokens
        
    def decode(self, token_ids):
        """Decode tokens back to text."""
        return self.tokenizer.decode(token_ids)


#####################################################################
#############             Model Configuration
#####################################################################            

class Llama3Model(nn.Module):
    DEFAULT_CONFIG = {
        "vocab_size": 128256,      # Vocabulary size
        "context_length": 4096,  # Context length (131072)
        "emb_dim": 2048,            # Embedding dimension
        "n_heads": 32,              # Number of attention heads
        "n_layers": 16,             # Number of layers
        "hidden_dim": 8192,         # Size of the intermediate dimension in FeedForward
        "n_kv_groups": 8,           # Key-Value groups for grouped-query attention
        "rope_base": 500_000.0,     # The base in RoPE's "theta"
        "dtype": torch.bfloat16,    # Lower-precision dtype to reduce memory usage
        "rope_freq": {              # RoPE frequency scaling
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_context_length": 8192,
        }
    }  
    
    _buffers_cache = {}  # Static cache for shared buffers
    
    def __init__(self, cfg=None):
        """Initialize a Llama3 model."""
        super().__init__()

        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Use default config if none provided
        self.cfg = cfg if cfg is not None else self.DEFAULT_CONFIG
        
        # Token embedding layer
        self.tok_emb = nn.Embedding(
            num_embeddings=self.cfg["vocab_size"],
            embedding_dim=self.cfg["emb_dim"],
            dtype=self.cfg["dtype"]
        )
        
        # Create transformer blocks with attention and feed-forward components
        self.trf_blocks = nn.ModuleList([
            self._create_transformer_block() for _ in range(self.cfg["n_layers"])
        ])
        
        # Final normalization and output projection
        self.final_norm = nn.RMSNorm(self.cfg["emb_dim"], eps=1e-5)
        self.out_head = nn.Linear(
            self.cfg["emb_dim"], 
            self.cfg["vocab_size"], 
            bias=False, 
            dtype=self.cfg["dtype"]
        )
        
        # Setup model and print initial metrics
        self._setup_and_print_metrics()

    def _setup_and_print_metrics(self):
        """Setup model and print initial metrics."""
        # Check buffer sharing
        print("\nChecking buffer sharing:")
        print(f"Mask shared: {self.trf_blocks[0].att.mask is self.trf_blocks[-1].att.mask}")
        print(f"Cos shared: {self.trf_blocks[0].att.cos is self.trf_blocks[-1].att.cos}")
        print(f"Sin shared: {self.trf_blocks[0].att.sin is self.trf_blocks[-1].att.sin}")
        
        # Calculate and print parameters
        total_params = sum(p.numel() for p in self.parameters())
        total_params_normalized = total_params - self.tok_emb.weight.numel()
        print(f"\nTotal number of parameters: {total_params:,}")
        print(f"Total number of unique parameters: {total_params_normalized:,}")
        
        # Calculate and print memory usage
        memory_fp32 = self._calculate_memory_size(torch.float32)
        memory_bf16 = self._calculate_memory_size(torch.bfloat16)
        print(f"\nMemory Usage:")
        print(f"float32 (PyTorch default): {memory_fp32:.2f} GB")
        print(f"bfloat16: {memory_bf16:.2f} GB")
        
        # Move to appropriate device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("\nUsing CUDA device")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("\nUsing MPS device")
        else:
            device = torch.device("cpu")
            print("\nUsing CPU device")
        
        self.to(device)

    def _calculate_memory_size(self, input_dtype=torch.float32):
        """Calculate model memory size for given dtype."""
        total_params = 0
        total_grads = 0
        for param in self.parameters():
            param_size = param.numel()
            total_params += param_size
            if param.requires_grad:
                total_grads += param_size
        
        total_buffers = sum(buf.numel() for buf in self.buffers())
        element_size = torch.tensor(0, dtype=input_dtype).element_size()
        total_memory_bytes = (total_params + total_grads + total_buffers) * element_size
        
        return total_memory_bytes / (1024**3)

    def _create_transformer_block(self):
        """Creates a transformer block with attention and feed-forward components."""
        attention = self._create_attention()
        ff = self._create_feed_forward()
        norm1 = nn.RMSNorm(self.cfg["emb_dim"], eps=1e-5)
        norm2 = nn.RMSNorm(self.cfg["emb_dim"], eps=1e-5)
        
        return nn.ModuleDict({
            'att': attention,
            'ff': ff,
            'norm1': norm1,
            'norm2': norm2
        })

    def _create_attention(self):
        """Creates a grouped query attention module."""
        head_dim = self.cfg["emb_dim"] // self.cfg["n_heads"]
        
        # Create attention components
        attention = nn.Module()
        attention.W_query = nn.Linear(self.cfg["emb_dim"], self.cfg["emb_dim"], bias=False, dtype=self.cfg["dtype"])
        attention.W_key = nn.Linear(self.cfg["emb_dim"], self.cfg["n_kv_groups"] * head_dim, bias=False, dtype=self.cfg["dtype"])
        attention.W_value = nn.Linear(self.cfg["emb_dim"], self.cfg["n_kv_groups"] * head_dim, bias=False, dtype=self.cfg["dtype"])
        attention.out_proj = nn.Linear(self.cfg["emb_dim"], self.cfg["emb_dim"], bias=False, dtype=self.cfg["dtype"])
        
        # Get shared buffers
        mask, cos, sin = self._get_attention_buffers(head_dim)
        attention.register_buffer("mask", mask)
        attention.register_buffer("cos", cos)
        attention.register_buffer("sin", sin)
        
        # Set attention attributes
        attention.num_heads = self.cfg["n_heads"]
        attention.num_kv_groups = self.cfg["n_kv_groups"]
        attention.head_dim = head_dim
        attention.group_size = self.cfg["n_heads"] // self.cfg["n_kv_groups"]
        
        # Add forward method
        def attention_forward(self, x):
            b, num_tokens, d_in = x.shape
            
            # Linear projections and reshape
            queries = self.W_query(x).view(b, num_tokens, attention.num_heads, attention.head_dim)
            keys = self.W_key(x).view(b, num_tokens, attention.num_kv_groups, attention.head_dim)
            values = self.W_value(x).view(b, num_tokens, attention.num_kv_groups, attention.head_dim)
            
            # Transpose for attention computation
            queries = queries.transpose(1, 2)
            keys = keys.transpose(1, 2)
            values = values.transpose(1, 2)
            
            # Apply RoPE
            keys = Llama3Model.compute_rope(keys, self.cos, self.sin)
            queries = Llama3Model.compute_rope(queries, self.cos, self.sin)
            
            # Expand keys and values for grouped attention
            keys = keys.repeat_interleave(attention.group_size, dim=1)
            values = values.repeat_interleave(attention.group_size, dim=1)
            
            # Compute attention scores
            attn_scores = queries @ keys.transpose(2, 3)
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
            attn_scores.masked_fill_(mask_bool, -torch.inf)
            
            # Compute attention weights with scaling
            scale = attention.head_dim ** -0.5
            attn_weights = torch.softmax(attn_scores * scale, dim=-1)
            
            # Compute context vectors
            context_vec = (attn_weights @ values).transpose(1, 2)
            context_vec = context_vec.reshape(b, num_tokens, d_in)
            context_vec = self.out_proj(context_vec)
            
            return context_vec
        
        attention.forward = attention_forward.__get__(attention)
        return attention

    def _get_attention_buffers(self, head_dim):
        """Get or create attention buffers."""
        key = (self.cfg["context_length"], head_dim, self.cfg["rope_base"],
               tuple(self.cfg["rope_freq"].values()) if self.cfg["rope_freq"] else None,
               self.cfg["dtype"])
        
        if key not in self._buffers_cache:
            mask = torch.triu(torch.ones(self.cfg["context_length"], self.cfg["context_length"]), diagonal=1)
            cos, sin = self._precompute_rope_params(head_dim)
            if self.cfg["dtype"] is not None:
                cos = cos.to(self.cfg["dtype"])
                sin = sin.to(self.cfg["dtype"])
            self._buffers_cache[key] = (mask, cos, sin)
        
        return self._buffers_cache[key]

    def _precompute_rope_params(self, head_dim):
        """Precompute RoPE parameters."""
        theta_base = self.cfg["rope_base"]
        freq_config = self.cfg["rope_freq"]
        context_length = self.cfg["context_length"]
        
        inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
        
        if freq_config is not None:
            low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
            high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]
            
            wavelen = 2 * math.pi / inv_freq
            inv_freq_llama = torch.where(
                wavelen > low_freq_wavelen,
                inv_freq / freq_config["factor"],
                inv_freq
            )
            
            smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
                freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
            )
            
            smoothed_inv_freq = (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
            is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
            inv_freq = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        
        positions = torch.arange(context_length)
        angles = positions[:, None] * inv_freq[None, :]
        angles = torch.cat([angles, angles], dim=1)
        
        return torch.cos(angles), torch.sin(angles)

    def _create_feed_forward(self):
        """Creates a feed-forward network module."""
        ff = nn.Module()
        ff.fc1 = nn.Linear(self.cfg["emb_dim"], self.cfg["hidden_dim"], dtype=self.cfg["dtype"], bias=False)
        ff.fc2 = nn.Linear(self.cfg["emb_dim"], self.cfg["hidden_dim"], dtype=self.cfg["dtype"], bias=False)
        ff.fc3 = nn.Linear(self.cfg["hidden_dim"], self.cfg["emb_dim"], dtype=self.cfg["dtype"], bias=False)
        
        def ff_forward(self, x):
            x_fc1 = self.fc1(x)
            x_fc2 = self.fc2(x)
            x = nn.functional.silu(x_fc1) * x_fc2
            return self.fc3(x)
        
        ff.forward = ff_forward.__get__(ff)
        return ff

    def forward(self, in_idx):
        """Forward pass of the model."""
        x = self.tok_emb(in_idx)
        
        for block in self.trf_blocks:
            # Attention with residual
            shortcut = x
            x = block['norm1'](x)
            x = block['att'](x.to(torch.bfloat16))
            x = x + shortcut
            
            # Feed-forward with residual
            shortcut = x
            x = block['norm2'](x)
            x = block['ff'](x.to(torch.bfloat16))
            x = x + shortcut
        
        x = self.final_norm(x)
        logits = self.out_head(x.to(torch.bfloat16))
        
        return logits

    @staticmethod
    def compute_rope(x, cos, sin):
        """Apply rotary position embeddings."""
        batch_size, num_heads, seq_len, head_dim = x.shape
        assert head_dim % 2 == 0, "Head dimension must be even"
        
        x1 = x[..., :head_dim // 2]
        x2 = x[..., head_dim // 2:]
        cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
        sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
        
        rotated = torch.cat((-x2, x1), dim=-1)
        x_rotated = (x * cos) + (rotated * sin)
        
        return x_rotated.to(dtype=x.dtype)

class Llama3Pipeline:
    def __init__(self, model, llama_size_str):
        """Initialize pipeline with model and HuggingFace hub tokenizer."""
        # Try to get token from environment first
        token = os.getenv("HUGGINGFACE_HUB_TOKEN")
        if not token:
            token = input("Enter your HuggingFace token: ")
        
        # Login with explicit token
        login(token=token)
        
        # Download tokenizer
        tokenizer_file_path = hf_hub_download(
            repo_id=f"meta-llama/Llama-3.2-{llama_size_str}-Instruct",
            filename="original/tokenizer.model",
            local_dir=f"Llama-3.2-{llama_size_str}-Instruct",
            token=token  # Explicitly pass token
        )   
        
        self.tokenizer = Tokenizer(tokenizer_file_path)
        self.chat_formatter = ChatFormat(self.tokenizer)
        self.model = model

    def text_to_token_ids(self, text):
        encoded = self.chat_formatter.encode(text)
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
        return encoded_tensor.to(self.model.device)

    def token_ids_to_text(self, token_ids):
        flat = token_ids.squeeze(0)  # remove batch dimension
        return self.tokenizer.decode(flat.tolist())

    def clean_text(self, text, header_end="assistant<|end_header_id|>\n\n"):
        index = text.find(header_end)
        if index != -1:
            return text[index + len(header_end):].strip()
        return text

    def generate(self, prompt, max_new_tokens, temperature=0.0, top_k=None):
        """Generate text using the provided model and tokenizer."""
        # Convert prompt to token ids
        idx = self.text_to_token_ids(prompt)
        eos_id = self.tokenizer.special_tokens["<|eot_id|>"]

        # Generate token by token
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.model.cfg["context_length"]:]
            with torch.no_grad():
                logits = self.model(idx_cond)
            logits = logits[:, -1, :]

            # Filter logits with top_k sampling
            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(logits < min_val, 
                                   torch.tensor(float('-inf')).to(self.model.device), 
                                   logits)

            # Apply temperature scaling
            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            # Stop if end-of-sequence token is encountered
            if idx_next.item() == eos_id:
                break

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

        # Convert to text and clean
        output_text = self.token_ids_to_text(idx)
        return self.clean_text(output_text)


#####################################################################
#############              Weight Initializer
#####################################################################  
  

class LlamaWeightLoader:
    def __init__(self, model, llama_size_str):
        """Initialize weight loader for a Llama3 model."""
        self.model = model
        self.param_config = model.cfg
        self.llama_size_str = llama_size_str
        self.device = model.device

    @staticmethod
    def assign(left, right, tensor_name="unknown"):
        """
        Assigns the right tensor to the left parameter with shape check.
        Returns a new torch.nn.Parameter.
        """
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")

        if isinstance(right, torch.Tensor):
            return torch.nn.Parameter(right.clone().detach())
        else:
            return torch.nn.Parameter(torch.tensor(right))

    def load_weights(self, params):
        """
        Loads weights into the model from the given parameter dictionary.
        """
        # Load token embedding weights
        self.model.tok_emb.weight = self.assign(
            self.model.tok_emb.weight, 
            params["model.embed_tokens.weight"], 
            "model.embed_tokens.weight"
        )

        # Load Transformer Block Weights
        for l in range(self.param_config["n_layers"]):
            self.load_transformer_block_weights(l, params)

        # Load final normalization and output layer weights
        self.model.final_norm.weight = self.assign(
            self.model.final_norm.weight, 
            params["model.norm.weight"], 
            "model.norm.weight"
        )
        self.load_output_layer_weights(params)

    def load_transformer_block_weights(self, layer_idx, params):
        """
        Loads weights for a specific transformer block layer.
        """
        # Load Attention Weights
        self.model.trf_blocks[layer_idx]['att'].W_query.weight = self.assign(
            self.model.trf_blocks[layer_idx]['att'].W_query.weight,
            params[f"model.layers.{layer_idx}.self_attn.q_proj.weight"],
            f"model.layers.{layer_idx}.self_attn.q_proj.weight"
        )
        self.model.trf_blocks[layer_idx]['att'].W_key.weight = self.assign(
            self.model.trf_blocks[layer_idx]['att'].W_key.weight,
            params[f"model.layers.{layer_idx}.self_attn.k_proj.weight"],
            f"model.layers.{layer_idx}.self_attn.k_proj.weight"
        )
        self.model.trf_blocks[layer_idx]['att'].W_value.weight = self.assign(
            self.model.trf_blocks[layer_idx]['att'].W_value.weight,
            params[f"model.layers.{layer_idx}.self_attn.v_proj.weight"],
            f"model.layers.{layer_idx}.self_attn.v_proj.weight"
        )
        self.model.trf_blocks[layer_idx]['att'].out_proj.weight = self.assign(
            self.model.trf_blocks[layer_idx]['att'].out_proj.weight,
            params[f"model.layers.{layer_idx}.self_attn.o_proj.weight"],
            f"model.layers.{layer_idx}.self_attn.o_proj.weight"
        )
        self.model.trf_blocks[layer_idx]['norm1'].weight = self.assign(
            self.model.trf_blocks[layer_idx]['norm1'].weight,
            params[f"model.layers.{layer_idx}.input_layernorm.weight"],
            f"model.layers.{layer_idx}.input_layernorm.weight"
        )

        # Load FeedForward Weights
        self.model.trf_blocks[layer_idx]['ff'].fc1.weight = self.assign(
            self.model.trf_blocks[layer_idx]['ff'].fc1.weight,
            params[f"model.layers.{layer_idx}.mlp.gate_proj.weight"],
            f"model.layers.{layer_idx}.mlp.gate_proj.weight"
        )
        self.model.trf_blocks[layer_idx]['ff'].fc2.weight = self.assign(
            self.model.trf_blocks[layer_idx]['ff'].fc2.weight,
            params[f"model.layers.{layer_idx}.mlp.up_proj.weight"],
            f"model.layers.{layer_idx}.mlp.up_proj.weight"
        )
        self.model.trf_blocks[layer_idx]['ff'].fc3.weight = self.assign(
            self.model.trf_blocks[layer_idx]['ff'].fc3.weight,
            params[f"model.layers.{layer_idx}.mlp.down_proj.weight"],
            f"model.layers.{layer_idx}.mlp.down_proj.weight"
        )
        self.model.trf_blocks[layer_idx]['norm2'].weight = self.assign(
            self.model.trf_blocks[layer_idx]['norm2'].weight,
            params[f"model.layers.{layer_idx}.post_attention_layernorm.weight"],
            f"model.layers.{layer_idx}.post_attention_layernorm.weight"
        )

    def load_output_layer_weights(self, params):
        """
        Loads weights for the output layer, with weight tying check.
        """
        if "lm_head.weight" in params.keys():
            self.model.out_head.weight = self.assign(
                self.model.out_head.weight, 
                params["lm_head.weight"], 
                "lm_head.weight"
            )
        else:
            self.model.out_head.weight = self.assign(
                self.model.out_head.weight, 
                params["model.embed_tokens.weight"], 
                "model.embed_tokens.weight"
            )
            print("Model uses weight tying.")
        
        print("Weight tying:", torch.equal(self.model.tok_emb.weight, self.model.out_head.weight))

    def download_and_load_weights(self):
        """
        Downloads and loads model weights into the model.
        """
        # Try to get token from environment first
        token = os.getenv("HUGGINGFACE_HUB_TOKEN")
        if not token:
            token = input("Enter your HuggingFace token: ")
        
        # Login with explicit token
        login(token=token)
        
        combined_weights = {}
        if self.llama_size_str == "1B":
            weights_file = hf_hub_download(
                repo_id=f"meta-llama/Llama-3.2-{self.llama_size_str}-Instruct",
                filename="model.safetensors",
                local_dir=f"Llama-3.2-{self.llama_size_str}-Instruct",
                token=token
            )
            combined_weights = self.load_file(weights_file)
        else:
            for i in range(1, 3):
                weights_file = hf_hub_download(
                    repo_id=f"meta-llama/Llama-3.2-{self.llama_size_str}-Instruct",
                    filename=f"model-0000{i}-of-00002.safetensors",
                    local_dir=f"Llama-3.2-{self.llama_size_str}-Instruct",
                    token=token
                )
                current_weights = self.load_file(weights_file)
                combined_weights.update(current_weights)

        self.load_weights(combined_weights)
        self.model.to(self.device)
        print("Weight tying:", torch.equal(self.model.tok_emb.weight, self.model.out_head.weight))        
        
        # Free up memory
        del combined_weights
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def load_file(filepath):
        """
        Loads a safetensors file and returns the state dictionary.
        """
        print(f"Loading weights from {filepath}")
        return load_file(filepath)  # Use safetensors.torch.load_file


# After your model initialization
torch.manual_seed(123)
model = Llama3Model()  # This will use the DEFAULT_CONFIG

# Initialize weight loader and load weights
loader = LlamaWeightLoader(model, "1B")
loader.download_and_load_weights()

# Then create the pipeline with this model
pipeline = Llama3Pipeline(model=model, llama_size_str="1B")

#####################################################################
#############             Inference 
#####################################################################  

def generate_text(prompt: str, pipeline) -> str:
    """Generates text using the given pipeline and prompt."""
    response = pipeline.generate(
        prompt=prompt,
        max_new_tokens=250,
        top_k=1,
        temperature=0.0
    )
    return response

# Example usage
response = generate_text("What is the capital of France?", pipeline)
print(f"Output text:\n{response}")

#####################################################################  


    

