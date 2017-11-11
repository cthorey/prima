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

|     bird |   blank |   cattle |   chimpanzee |   elephant |   forest buffalo |   gorilla |   hippopotamus |    human |      hyena |   large ungulate |   leopard |       lion |   other (non-primate) |   other (primate) |   pangolin |   porcupine |   reptile |   rodent |   small antelope |   small cat |   wild dog |   duiker |      hog |
|---------:|--------:|---------:|-------------:|-----------:|-----------------:|----------:|---------------:|---------:|-----------:|-----------------:|----------:|-----------:|----------------------:|------------------:|-----------:|------------:|----------:|---------:|-----------------:|------------:|-----------:|---------:|---------:|
| 0.299147 | 0.29883 |  0.10575 |     0.296021 |   0.239182 |       0.00539591 | 0.0617278 |      0.0573906 | 0.298831 | 0.00936577 |        0.0712738 | 0.0774029 | 0.00376589 |              0.298233 |          0.326893 |  0.0247637 |    0.146388 | 0.0084622 | 0.301484 |        0.0808171 |   0.0316565 |  0.0114814 | 0.307882 | 0.297421 |

## Validation set 

|     bird |    blank |   cattle |   chimpanzee |   elephant |   forest buffalo |   gorilla |   hippopotamus |    human |     hyena |   large ungulate |   leopard |       lion |   other (non-primate) |   other (primate) |   pangolin |   porcupine |    reptile |   rodent |   small antelope |   small cat |   wild dog |   duiker |     hog |
|---------:|---------:|---------:|-------------:|-----------:|-----------------:|----------:|---------------:|---------:|----------:|-----------------:|----------:|-----------:|----------------------:|------------------:|-----------:|------------:|-----------:|---------:|-----------------:|------------:|-----------:|---------:|--------:|
| 0.310993 | 0.323052 | 0.096924 |     0.327703 |   0.234581 |       0.00580096 | 0.0427606 |      0.0687436 | 0.295511 | 0.0159044 |        0.0722019 | 0.0757053 | 0.00300638 |              0.295152 |          0.357013 |  0.0271423 |    0.147834 | 0.00683391 | 0.290091 |        0.0785335 |   0.0131762 |  0.0111323 | 0.268739 | 0.29471 |



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