from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="facebook/sam2.1-hiera-large",
    filename="sam2.1_hiera_large.pt",
    local_dir="/workspace/checkpoints/segmentation/sam2",
)
print("done")
