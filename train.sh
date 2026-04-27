#  python train.py -s data/ref_nerf/coffee --eval --white_background   
#  python train.py -s data/ref_nerf/helmet --eval  --white_background  --lambda_normal_smooth 1.0
#  python train.py -s data/ref_nerf/ball --eval  --white_background --lambda_normal_smooth 1.0 
#  python train.py -s data/ref_nerf/teapot --eval  --white_background 
#  python train.py -s data/ref_nerf/toaster --eval  --white_background   
#  python train.py -s data/ref_nerf/car --eval  --white_background 

#  python train.py -s data/GlossySynthetic/angel_blender --eval --white_background   
#  python train.py -s data/GlossySynthetic/potion_blender --eval  --white_background   
#  python train.py -s data/GlossySynthetic/horse_blender --eval  --white_background   
#  python train.py -s data/GlossySynthetic/luyu_blender --eval  --white_background    
#  python train.py -s data/GlossySynthetic/teapot_blender --eval  --white_background 
#  python train.py -s data/GlossySynthetic/bell_blender --eval  --white_background   
#  python train.py -s data/GlossySynthetic/tbell_blender --eval  --white_background  --lambda_normal_smooth 1.0
#  python train.py -s data/GlossySynthetic/cat_blender --eval  --white_background 


 CUDA_VISIBLE_DEVICES=3 python train.py -s /nfs/508_users/disk5/wsq/ENVS/shadow_gaussian/data/ref-real/gardenspheres 
    --eval --iterations 20000 
    --indirect_from_iter 10000 
    --volume_render_until_iter 0 
    --initial 1 
    --init_until_iter 5000 
    --lambda_normal_smooth 0.45 
    -r 4 

#  python train.py -s data/ref_real/toycar --eval --iterations 20000 --indirect_from_iter 10000 --volume_render_until_iter 0  --initial 1 --init_until_iter 3000  -r 4
#  python train.py -s data/ref_real/sedan --eval --iterations 20000 --indirect_from_iter 10000 --volume_render_until_iter 0  --initial 1 --init_until_iter 3000  -r 8 

# python train.py -s blender_data_2 \
#     --eval \
#     --iterations 30000 \
#     --indirect_from_iter 10000 \
#     --volume_render_until_iter 0 \
#     --initial 1 \
#     --init_until_iter 3000 \
#     --lambda_normal_smooth 0.45 \
#     -r 4

# python train.py -s ../blender_data_2 \
#     --eval \
#     --iterations 20000 \
#     --indirect_from_iter 10000 \
#     --volume_render_until_iter 0 \
#     --initial 1 \
#     --init_until_iter 5000 \
#     --lambda_normal_smooth 0.45 \
#     -r 2

CUDA_VISIBLE_DEVICES=4  python train.py -s /nfs/508_users/disk5/wsq/ENVS/shadow_gaussian/data/blender/diffuse --eval --iterations 40001 --indirect_from_iter 7000 --volume_render_until_iter 0 --initial 1 --init_until_iter 3000 -r 4 --env_HDR --read_envmap

CUDA_VISIBLE_DEVICES=3  python train.py -s /nfs/508_users/disk5/wsq/ENVS/shadow_gaussian/data/blender/diffuse --eval --iterations 30001 --indirect_from_iter 10000 --volume_render_until_iter 0 --initial 1 --init_until_iter 5000 -r 2 --lambda_normal_smooth 0.45 

CUDA_VISIBLE_DEVICES=7  python train.py -s /nfs/508_users/disk5/wsq/ENVS/shadow_gaussian/data/blender/diffuse --eval --iterations 30001 --indirect_from_iter 10000 --volume_render_until_iter 0 --initial 1 --init_until_iter 5000 -r 4 --env_HDR --read_envmap --zero_metallic


CUDA_VISIBLE_DEVICES=7  python train.py -s /nfs/508_users/disk5/wsq/ENVS/shadow_gaussian/data/blender/table --eval --iterations 30001 --indirect_from_iter 10000 --volume_render_until_iter 0 --initial 1 --init_until_iter 5000 -r 4 --env_HDR --read_envmap



# debug
CUDA_VISIBLE_DEVICES=7 /nfs/508_users/disk5/wsq/miniconda3/envs/ref/bin/python debug_render.py   --model_path /nfs/508_users/disk5/wsq/ENVS/ref-xjm/output/table_reflect/table_reflect-0330_1108   --iteration 30001   --all_cameras --max_cameras 60

CUDA_VISIBLE_DEVICES=7 /nfs/508_users/disk5/wsq/miniconda3/envs/ref/bin/python /nfs/508_users/disk5/wsq/ENVS/ref-xjm/debug_render.py \
  --model_path /nfs/508_users/disk5/wsq/ENVS/ref-xjm/output/table_reflect/table_reflect-0330_1108 \
  --iteration 30000 \
  --split train \
  --reproduce_training_vis \
  --mesh_iteration 28000 \
  --all_cameras \
  --max_cameras 60

CUDA_VISIBLE_DEVICES=7 /nfs/508_users/disk5/wsq/miniconda3/envs/ref/bin/python /nfs/508_users/disk5/wsq/ENVS/ref-xjm/debug_render.py \
  --model_path /nfs/508_users/disk5/wsq/ENVS/ref-xjm/output/table_reflect/table_reflect-0407_1422 \
  --iteration 30000 \
  --split train \
  --reproduce_training_vis \
  --mesh_iteration 28000 \
  --all_cameras \
  --max_cameras 60 \
  --use_gt_normal_in_reshade

  
CUDA_VISIBLE_DEVICES=7 /nfs/508_users/disk5/wsq/miniconda3/envs/ref/bin/python /nfs/508_users/disk5/wsq/ENVS/ref-xjm/debug_render.py \
  --model_path /nfs/508_users/disk5/wsq/ENVS/ref-xjm/output/table_reflect/table_reflect-0413_1342_changeSpecularWeightNoUsingMetallic \
  --iteration 30000 \
  --split train \
  --reproduce_training_vis \
  --mesh_iteration 28000 \
  --all_cameras \
  --max_cameras 60 \
  --use_gt_normal_in_reshade


  CUDA_VISIBLE_DEVICES=5 /nfs/508_users/disk5/wsq/miniconda3/envs/ref/bin/python train.py -s /nfs/508_users/disk5/wsq/ENVS/shadow_gaussian/data/blender/table_video --eval --iterations 30001 --indirect_from_iter 10000 --volume_render_until_iter 0 --initial 1 --init_until_iter 5000 -r 4 --env_HDR --read_envmap --read_roughness --read_metallic
