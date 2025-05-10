from transformers import PreTrainedModel, AutoConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import AutoConfig, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
import torch
import torch.nn as nn

class TimesFMConfig(PretrainedConfig):
    model_type = "timesfm"
    
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=1024,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

class TimesFMModel(PreTrainedModel):
    config_class = TimesFMConfig
    base_model_prefix = "timesfm"
    supports_gradient_checkpointing = True
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # Embeddings
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Language model head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.init_weights()
        
        # Set generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=24,
            num_beams=4,
            temperature=0.7,
            do_sample=True,
            pad_token_id=config.pad_token_id,
            bos_token_id=config.pad_token_id,
            eos_token_id=config.pad_token_id
        )
    
    def get_input_embeddings(self):
        return self.embeddings
    
    def set_input_embeddings(self, value):
        self.embeddings = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        if position_ids is None:
            position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        
        # Get embeddings
        embeddings = self.embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        # Add position embeddings
        hidden_states = embeddings + position_embeddings
        
        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Get logits through LM head
        logits = self.lm_head(hidden_states)
        
        # Handle loss calculation if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        
        return {"loss": loss, "logits": logits} if loss is not None else logits
    
    def generate(self, input_ids, **kwargs):
        """Custom generate method for time series forecasting.
        
        Args:
            input_ids: Input tensor with shape [batch_size, seq_len, n_features]
            **kwargs: Additional keyword arguments for generation
        
        Returns:
            Generated predictions with shape [batch_size, seq_len+max_new_tokens, n_features]
        """
        # Move input to CPU to avoid MPS issues
        input_ids = input_ids.to('cpu')
        
        # Get dimensions
        if len(input_ids.shape) == 3:
            batch_size, seq_len, n_features = input_ids.size()
        else:
            # Handle case if input is [batch_size, seq_len]
            batch_size, seq_len = input_ids.size()
            n_features = 1
            input_ids = input_ids.unsqueeze(-1)
        
        # Determine how many new tokens to generate
        max_new_tokens = kwargs.get('max_new_tokens', self.generation_config.max_new_tokens)
        
        # For time series forecasting, we'll directly use the input features
        # without going through token embedding/de-embedding
        
        # Create a projection layer to map hidden states directly to features
        feature_projection = nn.Linear(self.config.hidden_size, n_features).to('cpu')
        
        # Initialize the predictions tensor
        predictions = torch.zeros(batch_size, max_new_tokens, n_features, device='cpu')
        
        # Initialize hidden states from input
        # Project input features to hidden dimension
        hidden_projection = nn.Linear(n_features, self.config.hidden_size).to('cpu')
        hidden_states = hidden_projection(input_ids)
        
        # Process through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # Auto-regressive generation
        current_input = input_ids.clone()
        
        for i in range(max_new_tokens):
            # Get the last timestep
            last_step = current_input[:, -1:, :]
            
            # Project to hidden dimension
            h = hidden_projection(last_step)
            
            # Process through transformer layers
            for layer in self.layers:
                h = layer(h)
            
            # Project directly to feature space
            next_step = feature_projection(h)
            
            # Store prediction
            predictions[:, i:i+1, :] = next_step
            
            # Update for next iteration
            if i < max_new_tokens - 1:
                current_input = torch.cat([current_input[:, 1:, :], next_step], dim=1)
        
        # Concatenate with input to create the full output sequence
        output = torch.cat([input_ids, predictions], dim=1)
        
        return output
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        input_shape = input_ids.shape
        
        # If past is used, only the last token should be processed
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
        }
    
    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layernorm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = nn.GELU()
    
    def forward(self, hidden_states, attention_mask=None):
        # Self-attention with residual connection and layer norm
        attention_output = self.attention(hidden_states, attention_mask)
        # Make sure we're returning just the hidden states, not a tuple
        if isinstance(attention_output, tuple):
            attention_output = attention_output[0]
        hidden_states = self.layernorm1(hidden_states + attention_output)
        
        # Feed-forward with residual connection and layer norm
        intermediate_output = self.intermediate(hidden_states)
        intermediate_output = self.activation(intermediate_output)
        
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        hidden_states = self.layernorm2(hidden_states + layer_output)
        
        return hidden_states

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.output = nn.Linear(self.all_head_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        # Linear projections
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Scale dot-product attention
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.attention_head_size, dtype=torch.float))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Softmax normalization
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Context aggregation
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Output projection
        output = self.output(context_layer)
        
        # Return only the output tensor
        return output

# Register the model and config
AutoConfig.register("timesfm", TimesFMConfig)
AutoModelForCausalLM.register(TimesFMConfig, TimesFMModel) 