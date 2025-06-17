#paragma once

#include <torch/torch.h>
#include <vector>
#include <tuple>
#include <cmath>

// utilities
torch::Tensor get_2d_sincos_pos_embed(
    int64_t embed_dim, int64_t grid_size, bool cls_token=false
);

torch::Tensor get_2d_sincos_pos_embed_from_grid(
    int64_t embed_dim, const torch::Tensor& grid
);

torch::Tensor get_1d_sincos_pos_embed_from_grid(
    int64_t embed_dim, const torch::Tensor& pos
);

// Patch Embedding model
class PatchEmbedImpl : public torch::nn::Module {
    public:
        // Constructor
        PatchEmbedImpl(
            int64_t img_size = 224, 
            int64_t patch_size=16, 
            int64_t in_chans = 3,
            int64_t embed_dim=768
        );
        // forward method
        torch::Tensor forward(const torch::Tensor& x);

        int64_t num_patches;
        std::pair<int64_t, int64_t> patch_size_tuple;

    private:
        torch::nn::Conv2d proj{nullptr}; // will be intialized later
};
TORCH_MODULE(PatchEmbed);

// MLP Module
class MlpImpl : public torch::nn::Module{
    public:
        // Constructor
        MlpImpl(
            int64_t in_features,
            int64_t hidden_features,
            int64_t out_features,
            float drop=0.
        );

        // Forward Method
        torch::Tensor forward(torch::Tensor x);

    private:
        torch::nn::Linear fc1{nullptr};
        torch::nn::Linear fc2{nullptr};
        torch::nn::GELU act{};
        torch::nn::Dropout drop1{nullptr};
        torch::nn::Dropout drop2{nullptr};
};
TORCH_MODULE(Mlp);

// Attention Module
class AttentionImpl : public torch::nn::Module{
    public:
        // Constructor
        AttentionImpl(
            int64_t dim,
            int64_t num_heads = 8,
            bool qkv_bias = false,
            float attn_drop = 0.,
            float proj_drop = 0.
        );
        // Forward method
        torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(Attention);

class BlockImpl : public torch::nn::Module{
    public:
        BlockImpl(
            int64_t dim,
            int64_t num_heads,
            double mlp_ratio = 4.,
            bool qkv_bias = false,
            float drop = 0.,
            float attn_drop = 0.
            float drop_path = 0.
        );
        // Forward Method
        torch::Tensor forward(torch::Tensor x);
    private:
        torch::nn::LayerNorm norm1{nullptr};
        Attention attn{nullptr};
        torch::nn::LayerNorm norm2{nullptr};
        Mlp mlp(nullptr);
        float drop_path_prob;
        torch::Tensor drop_path(const torch::Tensor& x);
};
TORCH_MODULE(Block);

// Masked AutoEncode ViT
class MaskedAutoencoderViTImpl : public torch::nn::Module{
    public:
        MaskedAutoencoderViTImpl(
            int64_t img_size = 224,
            int64_t patch_size = 16,
            int64_t in_chans = 3,
            int64_t embed_dim = 1024,
            int64_t depth = 24,
            int64_t num_heads = 16,
            int64_t decoder_embed_dim = 512,
            int64_t decoder_depth = 8,
            int64_t decoder_num_heads = 16,
            double mlp_ratio = 4.,
            bool norm_pix_loss = false
        );

        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
            const torch::Tensor& imgs,
            double mask_ratio = 0.75
        );

        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_encoder(
            const torch::Tensor& x,
            double mask_ratio
        );

        torch::Tensor forward_decoder(
            const torch::Tensor& x,
            const torch::Tensor& ids_restore
        );

        torch::Tensor forward_loss(
            const torch::Tensor& imgs,
            const torch::Tensor& pred,
            const torch::Tensor& mask
        )

        //Utility functions
        torch::Tensor patchify(const torch::Tensor& imgs);
        torch::Tensor unpatchify(const torch::Tensor& x);
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> random_masking(
            const torch::Tensor& x,
            double mask_ratio
        );
    
    private:
        void intialize_weights();

        // encoder
        PatchEmbed patch_embed{nullptr};
        torch::Tensor cls_token;
        torch::Tensor pos_embed;
        torch::nn::ModuleList blocks;
        torch::nn::LayerNorm norm{nullptr};

        //decoder
        torch::nn::Linear decoder_embed{nullptr};
        torch::Tensor mask_token;
        torch::Tensor decoder_pos_embed;
        torch::nn::ModuleList decoder_blocks;
        torch::nn::LayerNorm decoder_norm{nullptr};
        torch::nn::Linear decoder_pred{nullptr};

        bool norm_pix_loss;
        int64_t patch_size_int;
};
TORCH_MODULE(MaskedAutoencoderViT);

// Model Selection functions
MaskedAutoencoderViT mae_vit_base_patch16_dec512d8b(bool norm_pix_loss = false);
MaskedAutoencoderViT mae_vit_large_patch16_dec512d8b(bool norm_pix_loss = false);
MaskedAutoencoderViT mae_vit_huge_patch14_dec512d8b(bool norm_pix_loss = false);