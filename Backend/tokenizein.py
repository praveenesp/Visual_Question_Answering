def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionTransformer(nn.Module):
    def __init__(self, input_resolution, patch_size, width, layers, heads, embed_dim):
        super(VisionTransformer, self).__init__()
        
        # Calculate number of patches
        num_patches = (input_resolution // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2  # Assuming input images are RGB
        num_classes=100
        # Patch embedding layer
        self.patch_embedding = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional embeddings
        self.positional_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        
        # Transformer encoder layers
        self.transformer_encoder_layers = nn.ModuleList([
            torch._transformer_encoder_layer_fwd(embed_dim, heads, width) for _ in range(layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(embed_dim, num_classes)  # Assuming classification task
    
    def forward(self, x):
        # Patch embedding
        x = self.patch_embedding(x)
        
        # Flatten patches
        x = x.flatten(2).transpose(1, 2)
        
        # Add positional embeddings
        x = x + self.positional_embeddings
        
        # Transformer encoder layers
        for layer in self.transformer_encoder_layers:
            x = layer(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Output layer
        x = self.output_layer(x)
        
        return x
class YourClass:
    def __init__(self):
        
        
        def create_vision_transformer(self, image_resolution, vision_patch_size, vision_width, vision_layers, vision_heads, embed_dim):
            vision_model = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )
        
            return vision_model



def encode_image(self, image):
        return self.visual(image.type(self.dtype))

def encode_text(self, text):
    x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

    x = x + self.positional_embedding.type(self.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

    return x







    