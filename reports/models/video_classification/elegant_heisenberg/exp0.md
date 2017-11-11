# Best model 

- model task : video_classification 
- model name : elegant_heisenberg 
- data name : prima_d1_500 
- model description : benchmark 
- experiment : exp0 
- model folder : /workdir/models/video_classification/elegant_heisenberg 
- creation time : 11/11/2017 

# Best model - training summary 

## Training set 

|    bird |   blank |   cattle |   chimpanzee |   elephant |   forest buffalo |   gorilla |   hippopotamus |   human |      hyena |   large ungulate |   leopard |       lion |   other (non-primate) |   other (primate) |   pangolin |   porcupine |    reptile |   rodent |   small antelope |   small cat |   wild dog |   duiker |     hog |
|--------:|--------:|---------:|-------------:|-----------:|-----------------:|----------:|---------------:|--------:|-----------:|-----------------:|----------:|-----------:|----------------------:|------------------:|-----------:|------------:|-----------:|---------:|-----------------:|------------:|-----------:|---------:|--------:|
| 1.34539 | 1.40865 |  0.35569 |      1.41103 |    1.04565 |       0.00855555 |   0.17299 |       0.164464 |  1.4345 | 0.00752112 |         0.208233 |  0.200962 | 0.00215548 |               1.42599 |           1.47382 |  0.0598253 |    0.540478 | 0.00538409 |   1.4504 |         0.249976 |    0.080122 |  0.0203173 |  1.49526 | 1.42049 |

## Validation set 

|     bird |   blank |   cattle |   chimpanzee |   elephant |   forest buffalo |   gorilla |   hippopotamus |   human |     hyena |   large ungulate |   leopard |        lion |   other (non-primate) |   other (primate) |   pangolin |   porcupine |     reptile |   rodent |   small antelope |   small cat |   wild dog |   duiker |     hog |
|---------:|--------:|---------:|-------------:|-----------:|-----------------:|----------:|---------------:|--------:|----------:|-----------------:|----------:|------------:|----------------------:|------------------:|-----------:|------------:|------------:|---------:|-----------------:|------------:|-----------:|---------:|--------:|
| 0.701961 | 1.55829 | 0.318101 |      1.64683 |    1.02082 |       0.00972381 |  0.115427 |       0.201904 | 1.41646 | 0.0292103 |         0.211464 |  0.194096 | 0.000181274 |                1.4071 |           1.57162 |  0.0674969 |    0.549133 | 0.000411983 |  1.37708 |         0.241195 |   0.0196312 |  0.0194769 |  1.22514 | 1.39751 |



# Exepriment description 

```python 
{   'callback_config': {   'cycliclr': {   'base_lr': 0.0005,
                                           'max_lr': 0.006,
                                           'mode': 'triangular2',
                                           'step_size': 2000.0},
                           'keep_only_n': 5,
                           'mode': 'min',
                           'monitor': 'val_loss',
                           'patience': 35,
                           'reducelr': None,
                           'write_graph': True,
                           'write_images': False},
    'data_config': {   'base_augmentation': {   'rescale': 0.00392156862745098},
                       'batch_size': 2,
                       'mask_bcmode': None,
                       'mask_size': None,
                       'shuffle': True,
                       'target_size': <BoxList: [4, 128, 128, 3]>,
                       'test_augmentation': {   'rescale': 0.00392156862745098},
                       'train_aug_specific': {   'fill_mode': 'nearest',
                                                 'height_shift_range': 0.05,
                                                 'horizontal_flip': True,
                                                 'rotation_range': 15,
                                                 'shear_range': 0.1,
                                                 'vertical_flip': True,
                                                 'width_shift_range': 0.05,
                                                 'zoom_range': 0.1},
                       'train_augmentation': {   'fill_mode': 'nearest',
                                                 'height_shift_range': 0.05,
                                                 'horizontal_flip': True,
                                                 'rescale': 0.00392156862745098,
                                                 'rotation_range': 15,
                                                 'shear_range': 0.1,
                                                 'vertical_flip': True,
                                                 'width_shift_range': 0.05,
                                                 'zoom_range': 0.1},
                       'train_batch_index': 0,
                       'train_batch_size': 2,
                       'train_directory': '/workdir/data/interim/prima_d1_500',
                       'train_height': 224,
                       'train_n': 14870,
                       'train_nb_sample': 14870,
                       'train_num_classes': 24,
                       'train_num_frames': 16,
                       'train_shuffle': True,
                       'train_target_size': <BoxList: [4, 128, 128, 3]>,
                       'train_total_batches_seen': 0,
                       'train_width': 224,
                       'validation_augmentation': {   'rescale': 0.00392156862745098},
                       'validation_batch_index': 0,
                       'validation_batch_size': 2,
                       'validation_directory': '/workdir/data/interim/prima_d1_500',
                       'validation_height': 224,
                       'validation_n': 1659,
                       'validation_nb_sample': 1659,
                       'validation_num_classes': 24,
                       'validation_num_frames': 16,
                       'validation_shuffle': True,
                       'validation_target_size': <BoxList: [4, 128, 128, 3]>,
                       'validation_total_batches_seen': 0,
                       'validation_width': 224},
    'data_folder': '/workdir/data/interim/prima_d1_500',
    'data_name': 'prima_d1_500',
    'expname': 'exp0',
    'finetunning': None,
    'fit_config': <Box: {'steps_per_epoch': 150, 'epochs': 1}>,
    'input_shape': <BoxList: [16, 224, 224, 3]>,
    'model_description': 'benchmark',
    'model_folder': '/workdir/models/video_classification/elegant_heisenberg',
    'model_name': 'elegant_heisenberg',
    'model_task': 'video_classification',
    'stage': 'training',
    'training_config': {   'finetunning': None,
                           'loss': 'binary_crossentropy',
                           'metrics': <BoxList: ['binary_crossentropy']>,
                           'model_config': {   'input_shape': <BoxList: [4, 128, 128, 3]>,
                                               'last_layer': None,
                                               'mask_shape': None,
                                               'num_classes': 24},
                           'optimizer': 'Adam',
                           'optimizer_config': {   'decay': 0.05,
                                                   'lr': 0.001}}}
```