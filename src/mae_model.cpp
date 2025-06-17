#include "mae_model.hpp"
#include <cmath>
#include <algorithm>

// Position embedding utilities
torch::Tensor get_1d_sincos_pos_embed_from_grid(int64_t embed_dim, const torch::Tensor& pos) {
    assert(embed_dim % 2 == 0);
    
    auto omega = torch::arange(embed_dim / 2, torch::dtype(torch::kFloat32));
    omega /= embed_dim / 2.;
    omega = 1. / torch::pow(10000, omega);
    
    auto pos_reshaped = pos.reshape({-1});
    auto out = torch::einsum("m,d->md", {pos_reshaped, omega});
    
    auto emb_sin = torch::sin(out);
    auto emb_cos = torch::cos(out);
    
    auto emb = torch::cat({emb_sin, emb_cos}, 1);
    return emb;
}

torch::Tensor get_2d_sincos_pos_embed_from_grid(int64_t embed_dim, const torch::Tensor& grid) {
    assert(embed_dim % 2 == 0);
    
    auto emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim / 2, grid[0]);
    auto emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim / 2, grid[1]);
    
    auto emb = torch::cat({emb_h, emb_w}, 1);
    return emb;
}

torch::Tensor get_2d_sincos_pos_embed(int64_t embed_dim, int64_t grid_size, bool cls_token) {
    auto grid_h = torch::arange(grid_size, torch::dtype(torch::kFloat32));
    auto grid_w = torch::arange(grid_size, torch::dtype(torch::kFloat32));
    auto grid = torch::meshgrid({grid_w, grid_h}, "xy");
    auto grid_stacked = torch::stack(grid, 0);
    
    grid_stacked = grid_stacked.reshape({2, 1, grid_size, grid_size});
    auto pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid_stacked);
    
    if (cls_token) {
        auto cls_pos_embed = torch::zeros({1, embed_dim});
        pos_embed = torch::cat({cls_pos_embed, pos_embed}, 0);
    }
    
    return pos_embed;
}

// PatchEmbed implementation
PatchEmbedImpl::PatchEmbedImpl(int64_t img_size, int64_t patch_size, int64_t in_chans, int64_t embed_dim) {
    patch_size_tuple = std::make_pair(patch_size, patch_size);
    auto grid_size = img_size / patch_size;
    num_patches = grid_size * grid_size;
    
    proj = register_module("proj", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_chans, embed_dim, patch_size).stride(patch_size)));
}

torch::Tensor PatchEmbedImpl::forward(const torch::Tensor& x) {
    auto B = x.size(0);
    auto out = proj(x);
    out = out.flatten(2).transpose(1, 2);
    return out;
}

// MLP implementation
MlpImpl::MlpImpl(int64_t in_features, int64_t hidden_features, int64_t out_features, float drop) {
    fc1 = register_module("fc1", torch::nn::Linear(in_features, hidden_features));
    fc2 = register_module("fc2", torch::nn::Linear(hidden_features, out_features));
    drop1 = register_module("drop1", torch::nn::Dropout(drop));
    drop2 = register_module("drop2", torch::nn::Dropout(drop));
}

torch::Tensor MlpImpl::forward(torch::Tensor x) {
    x = fc1(x);
    x = act(x);
    x = drop1(x);
    x = fc2(x);
    x = drop2(x);
    return x;
}

// Attention implementation
AttentionImpl::AttentionImpl(int64_t dim, int64_t num_heads, bool qkv_bias, float attn_drop_p, float proj_drop_p) 
    : num_heads(num_heads) {
    scale = 1.0 / std::sqrt(static_cast<double>(dim / num_heads));
    
    qkv = register_module("qkv", torch::nn::Linear(torch::nn::LinearOptions(dim, dim * 3).bias(qkv_bias)));
    attn_drop = register_module("attn_drop", torch::nn::Dropout(attn_drop_p));
    proj = register_module("proj", torch::nn::Linear(dim, dim));
    proj_drop = register_module("proj_drop", torch::nn::Dropout(proj_drop_p));
}

torch::Tensor AttentionImpl::forward(torch::Tensor x) {
    auto B = x.size(0);
    auto N = x.size(1);
    auto C = x.size(2);
    
    auto qkv_out = qkv(x).reshape({B, N, 3, num_heads, C / num_heads}).permute({2, 0, 3, 1, 4});
    auto q = qkv_out[0];
    auto k = qkv_out[1];
    auto v = qkv_out[2];
    
    auto attn = torch::matmul(q, k.transpose(-2, -1)) * scale;
    attn = attn.softmax(-1);
    attn = attn_drop(attn);
    
    x = torch::matmul(attn, v).transpose(1, 2).reshape({B, N, C});
    x = proj(x);
    x = proj_drop(x);
    
    return x;
}

// Block implementation
BlockImpl::BlockImpl(int64_t dim, int64_t num_heads, double mlp_ratio, bool qkv_bias, 
                     float drop, float attn_drop, float drop_path)
    : drop_path_prob(drop_path) {
    norm1 = register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim}).eps(1e-6)));
    attn = register_module("attn", Attention(dim, num_heads, qkv_bias, attn_drop, drop));
    norm2 = register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim}).eps(1e-6)));
    
    auto mlp_hidden_dim = static_cast<int64_t>(dim * mlp_ratio);
    mlp = register_module("mlp", Mlp(dim, mlp_hidden_dim, dim, drop));
}

torch::Tensor BlockImpl::drop_path(const torch::Tensor& x) {
    if (drop_path_prob == 0. || !is_training()) {
        return x;
    }
    
    auto keep_prob = 1 - drop_path_prob;
    auto shape = std::vector<int64_t>{x.size(0)};
    for (int i = 1; i < x.dim(); ++i) {
        shape.push_back(1);
    }
    
    auto random_tensor = keep_prob + torch::rand(shape, x.options());
    random_tensor = random_tensor.floor();
    auto output = x.div(keep_prob) * random_tensor;
    
    return output;
}

torch::Tensor BlockImpl::forward(torch::Tensor x) {
    x = x + drop_path(attn(norm1(x)));
    x = x + drop_path(mlp(norm2(x)));
    return x;
}

// MaskedAutoencoderViT implementation
MaskedAutoencoderViTImpl::MaskedAutoencoderViTImpl(
    int64_t img_size,
    int64_t patch_size,
    int64_t in_chans,
    int64_t embed_dim,
    int64_t depth,
    int64_t num_heads,
    int64_t decoder_embed_dim,
    int64_t decoder_depth,
    int64_t decoder_num_heads,
    double mlp_ratio,
    bool norm_pix_loss) 
    : norm_pix_loss(norm_pix_loss), patch_size_int(patch_size) {
    
    // Encoder
    patch_embed = register_module("patch_embed", PatchEmbed(img_size, patch_size, in_chans, embed_dim));
    auto num_patches = patch_embed->num_patches;
    
    cls_token = register_parameter("cls_token", torch::zeros({1, 1, embed_dim}));
    pos_embed = register_parameter("pos_embed", torch::zeros({1, num_patches + 1, embed_dim}), false);
    
    blocks = register_module("blocks", torch::nn::ModuleList());
    for (int64_t i = 0; i < depth; ++i) {
        blocks->push_back(Block(embed_dim, num_heads, mlp_ratio, true));
    }
    norm = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim}).eps(1e-6)));
    
    // Decoder
    decoder_embed = register_module("decoder_embed", torch::nn::Linear(embed_dim, decoder_embed_dim));
    mask_token = register_parameter("mask_token", torch::zeros({1, 1, decoder_embed_dim}));
    decoder_pos_embed = register_parameter("decoder_pos_embed", 
                                         torch::zeros({1, num_patches + 1, decoder_embed_dim}), false);
    
    decoder_blocks = register_module("decoder_blocks", torch::nn::ModuleList());
    for (int64_t i = 0; i < decoder_depth; ++i) {
        decoder_blocks->push_back(Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, true));
    }
    
    decoder_norm = register_module("decoder_norm", 
                                 torch::nn::LayerNorm(torch::nn::LayerNormOptions({decoder_embed_dim}).eps(1e-6)));
    decoder_pred = register_module("decoder_pred", 
                                 torch::nn::Linear(decoder_embed_dim, patch_size * patch_size * in_chans));
    
    initialize_weights();
}

void MaskedAutoencoderViTImpl::initialize_weights() {
    // Initialize positional embeddings
    auto pos_embed_val = get_2d_sincos_pos_embed(pos_embed.size(-1), 
                                                static_cast<int64_t>(std::sqrt(patch_embed->num_patches)), true);
    pos_embed.data().copy_(pos_embed_val.unsqueeze(0));
    
    auto decoder_pos_embed_val = get_2d_sincos_pos_embed(decoder_pos_embed.size(-1), 
                                                        static_cast<int64_t>(std::sqrt(patch_embed->num_patches)), true);
    decoder_pos_embed.data().copy_(decoder_pos_embed_val.unsqueeze(0));
    
    // Initialize patch_embed like nn.Linear
    auto w = patch_embed->named_parameters()["proj.weight"];
    torch::nn::init::xavier_uniform_(w.view({w.size(0), -1}));
    
    // Initialize cls_token and mask_token
    torch::nn::init::normal_(cls_token, 0, 0.02);
    torch::nn::init::normal_(mask_token, 0, 0.02);
    
    // Initialize Linear layers and LayerNorm
    for (auto& module : modules(false)) {
        if (auto* linear = module->as<torch::nn::Linear>()) {
            torch::nn::init::xavier_uniform_(linear->weight);
            if (linear->options.bias()) {
                torch::nn::init::constant_(linear->bias, 0);
            }
        } else if (auto* layer_norm = module->as<torch::nn::LayerNorm>()) {
            torch::nn::init::constant_(layer_norm->bias, 0);
            torch::nn::init::constant_(layer_norm->weight, 1.0);
        }
    }
}

torch::Tensor MaskedAutoencoderViTImpl::patchify(const torch::Tensor& imgs) {
    auto p = patch_size_int;
    assert(imgs.size(2) == imgs.size(3) && imgs.size(2) % p == 0);
    
    auto h = imgs.size(2) / p;
    auto w = imgs.size(3) / p;
    auto x = imgs.reshape({imgs.size(0), 3, h, p, w, p});
    x = torch::einsum("nchpwq->nhwpqc", {x});
    x = x.reshape({imgs.size(0), h * w, p * p * 3});
    
    return x;
}

torch::Tensor MaskedAutoencoderViTImpl::unpatchify(const torch::Tensor& x) {
    auto p = patch_size_int;
    auto h = static_cast<int64_t>(std::sqrt(x.size(1)));
    auto w = h;
    assert(h * w == x.size(1));
    
    auto x_reshaped = x.reshape({x.size(0), h, w, p, p, 3});
    x_reshaped = torch::einsum("nhwpqc->nchpwq", {x_reshaped});
    auto imgs = x_reshaped.reshape({x.size(0), 3, h * p, w * p});
    
    return imgs;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> MaskedAutoencoderViTImpl::random_masking(
    const torch::Tensor& x, double mask_ratio) {
    auto N = x.size(0);
    auto L = x.size(1);
    auto D = x.size(2);
    auto len_keep = static_cast<int64_t>(L * (1 - mask_ratio));
    
    auto noise = torch::rand({N, L}, x.options());
    
    auto ids_shuffle = torch::argsort(noise, 1, /*descending=*/false);
    auto ids_restore = torch::argsort(ids_shuffle, 1, /*descending=*/false);
    
    auto ids_keep = ids_shuffle.slice(1, 0, len_keep);
    auto x_masked = torch::gather(x, 1, ids_keep.unsqueeze(-1).repeat({1, 1, D}));
    
    auto mask = torch::ones({N, L}, x.options());
    mask.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, len_keep)}, 0);
    mask = torch::gather(mask, 1, ids_restore);
    
    return std::make_tuple(x_masked, mask, ids_restore);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> MaskedAutoencoderViTImpl::forward_encoder(
    const torch::Tensor& x, double mask_ratio) {
    // Embed patches
    auto x_embed = patch_embed(x);
    
    // Add pos embed w/o cls token
    x_embed = x_embed + pos_embed.slice(1, 1);
    
    // Masking
    auto [x_masked, mask, ids_restore] = random_masking(x_embed, mask_ratio);
    
    // Append cls token
    auto cls_tokens = cls_token + pos_embed.slice(1, 0, 1);
    cls_tokens = cls_tokens.expand({x_masked.size(0), -1, -1});
    x_masked = torch::cat({cls_tokens, x_masked}, 1);
    
    // Apply Transformer blocks
    for (const auto& blk : *blocks) {
        x_masked = blk->as<Block>()->forward(x_masked);
    }
    x_masked = norm(x_masked);
    
    return std::make_tuple(x_masked, mask, ids_restore);
}

torch::Tensor MaskedAutoencoderViTImpl::forward_decoder(const torch::Tensor& x, const torch::Tensor& ids_restore) {
    // Embed tokens
    auto x_decoded = decoder_embed(x);
    
    // Append mask tokens to sequence
    auto mask_tokens = mask_token.repeat({x_decoded.size(0), ids_restore.size(1) + 1 - x_decoded.size(1), 1});
    auto x_ = torch::cat({x_decoded.slice(1, 1), mask_tokens}, 1);
    x_ = torch::gather(x_, 1, ids_restore.unsqueeze(-1).repeat({1, 1, x_.size(2)}));
    x_decoded = torch::cat({x_decoded.slice(1, 0, 1), x_}, 1);
    
    // Add pos embed
    x_decoded = x_decoded + decoder_pos_embed;
    
    // Apply Transformer blocks
    for (const auto& blk : *decoder_blocks) {
        x_decoded = blk->as<Block>()->forward(x_decoded);
    }
    x_decoded = decoder_norm(x_decoded);
    
    // Predictor projection
    x_decoded = decoder_pred(x_decoded);
    
    // Remove cls token
    x_decoded = x_decoded.slice(1, 1);
    
    return x_decoded;
}

torch::Tensor MaskedAutoencoderViTImpl::forward_loss(const torch::Tensor& imgs, const torch::Tensor& pred, 
                                                   const torch::Tensor& mask) {
    auto target = patchify(imgs);
    
    // Debug shapes
    if (target.size(1) != pred.size(1) || target.size(2) != pred.size(2)) {
        std::cerr << "Shape mismatch in forward_loss:" << std::endl;
        std::cerr << "  target shape: " << target.sizes() << std::endl;
        std::cerr << "  pred shape: " << pred.sizes() << std::endl;
        std::cerr << "  mask shape: " << mask.sizes() << std::endl;
    }
    
    if (norm_pix_loss) {
        auto mean = target.mean(-1, true);
        auto var = target.var(-1, true);
        target = (target - mean) / (var + 1e-6).sqrt();
    }
    
    auto loss = (pred - target).pow(2);
    loss = loss.mean(-1);
    
    loss = (loss * mask).sum() / mask.sum();
    return loss;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> MaskedAutoencoderViTImpl::forward(
    const torch::Tensor& imgs, double mask_ratio) {
    auto [latent, mask, ids_restore] = forward_encoder(imgs, mask_ratio);
    auto pred = forward_decoder(latent, ids_restore);
    auto loss = forward_loss(imgs, pred, mask);
    
    return std::make_tuple(loss, pred, mask);
}

// Model factory functions
MaskedAutoencoderViT mae_vit_base_patch16_dec512d8b(bool norm_pix_loss) {
    return MaskedAutoencoderViT(224, 16, 3, 768, 12, 12, 512, 8, 16, 4., norm_pix_loss);
}

MaskedAutoencoderViT mae_vit_large_patch16_dec512d8b(bool norm_pix_loss) {
    return MaskedAutoencoderViT(224, 16, 3, 1024, 24, 16, 512, 8, 16, 4., norm_pix_loss);
}

MaskedAutoencoderViT mae_vit_huge_patch14_dec512d8b(bool norm_pix_loss) {
    return MaskedAutoencoderViT(224, 14, 3, 1280, 32, 16, 512, 8, 16, 4., norm_pix_loss);
}