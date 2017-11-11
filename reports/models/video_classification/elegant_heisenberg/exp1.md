# Best model 

- model task : video_classification 
- model name : elegant_heisenberg 
- data name : prima_d1_500 
- model description : benchmark 
- experiment : exp1 
- model folder : /workdir/models/video_classification/elegant_heisenberg 
- creation time : 11/11/2017 

# Best model - training summary 

## Training set 

|     bird |    blank |   cattle |   chimpanzee |   elephant |   forest buffalo |   gorilla |   hippopotamus |    human |      hyena |   large ungulate |   leopard |       lion |   other (non-primate) |   other (primate) |   pangolin |   porcupine |    reptile |   rodent |   small antelope |   small cat |   wild dog |   duiker |      hog |
|---------:|---------:|---------:|-------------:|-----------:|-----------------:|----------:|---------------:|---------:|-----------:|-----------------:|----------:|-----------:|----------------------:|------------------:|-----------:|------------:|-----------:|---------:|-----------------:|------------:|-----------:|---------:|---------:|
| 0.298967 | 0.295233 | 0.105645 |      0.29534 |   0.239049 |       0.00474386 | 0.0592036 |      0.0567576 | 0.298254 | 0.00452839 |        0.0688121 | 0.0668119 | 0.00160183 |              0.297502 |          0.304508 |  0.0244863 |      0.1463 | 0.00376732 |  0.30099 |        0.0797228 |   0.0312323 |  0.0100405 | 0.307329 | 0.296755 |

## Validation set 

|     bird |    blank |    cattle |   chimpanzee |   elephant |   forest buffalo |   gorilla |   hippopotamus |    human |     hyena |   large ungulate |   leopard |        lion |   other (non-primate) |   other (primate) |   pangolin |   porcupine |    reptile |   rodent |   small antelope |   small cat |   wild dog |   duiker |      hog |
|---------:|---------:|----------:|-------------:|-----------:|-----------------:|----------:|---------------:|---------:|----------:|-----------------:|----------:|------------:|----------------------:|------------------:|-----------:|------------:|-----------:|---------:|-----------------:|------------:|-----------:|---------:|---------:|
| 0.310543 | 0.315895 | 0.0964388 |     0.328797 |   0.234565 |       0.00518626 | 0.0427158 |      0.0671904 | 0.294982 | 0.0130813 |        0.0696206 | 0.0644798 | 0.000628077 |              0.294179 |          0.327455 |  0.0269548 |    0.147719 | 0.00165471 | 0.290163 |        0.0772138 |     0.01137 | 0.00964204 | 0.269925 | 0.294178 |



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
                       'batch_size': 5,
                       'mask_bcmode': None,
                       'mask_size': None,
                       'shuffle': True,
                       'target_size': <BoxList: [6, 128, 128, 3]>,
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
                       'train_batch_size': 5,
                       'train_directory': '/workdir/data/interim/prima_d1_500',
                       'train_height': 224,
                       'train_n': 14870,
                       'train_nb_sample': 14870,
                       'train_num_classes': 24,
                       'train_num_frames': 16,
                       'train_shuffle': True,
                       'train_target_size': <BoxList: [6, 128, 128, 3]>,
                       'train_total_batches_seen': 0,
                       'train_width': 224,
                       'validation_augmentation': {   'rescale': 0.00392156862745098},
                       'validation_batch_index': 0,
                       'validation_batch_size': 5,
                       'validation_directory': '/workdir/data/interim/prima_d1_500',
                       'validation_height': 224,
                       'validation_n': 1659,
                       'validation_nb_sample': 1659,
                       'validation_num_classes': 24,
                       'validation_num_frames': 16,
                       'validation_shuffle': True,
                       'validation_target_size': <BoxList: [6, 128, 128, 3]>,
                       'validation_total_batches_seen': 0,
                       'validation_width': 224},
    'data_folder': '/workdir/data/interim/prima_d1_500',
    'data_name': 'prima_d1_500',
    'expname': 'exp1',
    'finetunning': None,
    'fit_config': <Box: {'steps_per_epoch': 150, 'epochs': 150}>,
    'input_shape': <BoxList: [16, 224, 224, 3]>,
    'model_description': 'benchmark',
    'model_folder': '/workdir/models/video_classification/elegant_heisenberg',
    'model_name': 'elegant_heisenberg',
    'model_task': 'video_classification',
    'stage': 'training',
    'training_config': {   'finetunning': None,
                           'loss': 'binary_crossentropy',
                           'metrics': <BoxList: ['binary_crossentropy']>,
                           'model_config': {   'input_shape': <BoxList: [6, 128, 128, 3]>,
                                               'last_layer': None,
                                               'mask_shape': None,
                                               'num_classes': 24},
                           'optimizer': 'Adam',
                           'optimizer_config': {   'decay': 0.05,
                                                   'lr': 0.001}}}
```