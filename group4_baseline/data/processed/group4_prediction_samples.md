# Group4 Smoke Prediction Comparison

- manifest: `/root/final_project/group4_baseline/artifacts/peft_smoke/stage1_manifest_smoke_group4.json`
- samples: 5

## lora_lora-qv_target-qv_rows-64_seed-42

- method: lora
- lora_variant: qv
- target_modules: qv
- avg_loss: 3.0750
- avg_token_accuracy: 0.4760

### sample 0
- loss: 3.1285
- token_accuracy: 0.5455
- vision_path: `/root/final_project/group1_baseline/data/processed/clip_embeddings/000000203564.npy`
- label_text: ` A bicycle replica with a clock as the front wheel.`
- pred_text: ` A man with of a bike on the front wheel.`

### sample 1
- loss: 2.5845
- token_accuracy: 0.5000
- vision_path: `/root/final_project/group1_baseline/data/processed/clip_embeddings/000000322141.npy`
- label_text: ` A room with blue walls and a white sink and door.`
- pred_text: ` A man with a and and a white bed. toilet.`

### sample 2
- loss: 3.5852
- token_accuracy: 0.4615
- vision_path: `/root/final_project/group1_baseline/data/processed/clip_embeddings/000000016977.npy`
- label_text: ` A car that seems to be parked illegally behind a legally parked car`
- pred_text: ` A man driving is to be driving in in a row parked car`

### sample 3
- loss: 2.2694
- token_accuracy: 0.4444
- vision_path: `/root/final_project/group1_baseline/data/processed/clip_embeddings/000000106140.npy`
- label_text: ` A large passenger airplane flying through the air.`
- pred_text: ` A car bathroom plane flying over the clouds.`

### sample 4
- loss: 3.8075
- token_accuracy: 0.4286
- vision_path: `/root/final_project/group1_baseline/data/processed/clip_embeddings/000000106140.npy`
- label_text: ` There is a GOL plane taking off in a partly cloudy sky.`
- pred_text: ` A is a cariletDF in off from the field cloudy sky.`

## selective_ft_lora-na_target-qv_rows-64_seed-42

- method: selective_ft
- lora_variant: None
- target_modules: qv
- avg_loss: 3.0690
- avg_token_accuracy: 0.4760

### sample 0
- loss: 3.0977
- token_accuracy: 0.5455
- vision_path: `/root/final_project/group1_baseline/data/processed/clip_embeddings/000000203564.npy`
- label_text: ` A bicycle replica with a clock as the front wheel.`
- pred_text: ` A car with of a bike on the front wheel.`

### sample 1
- loss: 2.5724
- token_accuracy: 0.5000
- vision_path: `/root/final_project/group1_baseline/data/processed/clip_embeddings/000000322141.npy`
- label_text: ` A room with blue walls and a white sink and door.`
- pred_text: ` A woman with a and and a white bed. toilet.`

### sample 2
- loss: 3.5652
- token_accuracy: 0.4615
- vision_path: `/root/final_project/group1_baseline/data/processed/clip_embeddings/000000016977.npy`
- label_text: ` A car that seems to be parked illegally behind a legally parked car`
- pred_text: ` A man driving is to be driving in in a row parked car`

### sample 3
- loss: 2.3273
- token_accuracy: 0.4444
- vision_path: `/root/final_project/group1_baseline/data/processed/clip_embeddings/000000106140.npy`
- label_text: ` A large passenger airplane flying through the air.`
- pred_text: ` A car bathroom plane flying over the clouds.`

### sample 4
- loss: 3.7826
- token_accuracy: 0.4286
- vision_path: `/root/final_project/group1_baseline/data/processed/clip_embeddings/000000106140.npy`
- label_text: ` There is a GOL plane taking off in a partly cloudy sky.`
- pred_text: ` A is a cariletDF in off from the field cloudy sky.`

## lora_lora-all_weights_target-all_rows-64_seed-42

- method: lora
- lora_variant: all_weights
- target_modules: all
- avg_loss: 3.0689
- avg_token_accuracy: 0.4760

### sample 0
- loss: 3.1529
- token_accuracy: 0.5455
- vision_path: `/root/final_project/group1_baseline/data/processed/clip_embeddings/000000203564.npy`
- label_text: ` A bicycle replica with a clock as the front wheel.`
- pred_text: ` A man with of a bike on the front wheel.`

### sample 1
- loss: 2.5801
- token_accuracy: 0.5000
- vision_path: `/root/final_project/group1_baseline/data/processed/clip_embeddings/000000322141.npy`
- label_text: ` A room with blue walls and a white sink and door.`
- pred_text: ` A man with a and and a white bed. toilet.`

### sample 2
- loss: 3.5777
- token_accuracy: 0.4615
- vision_path: `/root/final_project/group1_baseline/data/processed/clip_embeddings/000000016977.npy`
- label_text: ` A car that seems to be parked illegally behind a legally parked car`
- pred_text: ` A man driving is to be driving in on a building parked car`

### sample 3
- loss: 2.2351
- token_accuracy: 0.4444
- vision_path: `/root/final_project/group1_baseline/data/processed/clip_embeddings/000000106140.npy`
- label_text: ` A large passenger airplane flying through the air.`
- pred_text: ` A man bathroom plane flying over the clouds.`

### sample 4
- loss: 3.7987
- token_accuracy: 0.4286
- vision_path: `/root/final_project/group1_baseline/data/processed/clip_embeddings/000000106140.npy`
- label_text: ` There is a GOL plane taking off in a partly cloudy sky.`
- pred_text: ` A is a cariletDF in off from the field cloudy sky.`
