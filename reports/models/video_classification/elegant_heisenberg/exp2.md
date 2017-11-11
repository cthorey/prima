# Best model 

- model task : video_classification 
- model name : elegant_heisenberg 
- data name : prima_d1_500 
- model description : benchmark 
- experiment : exp2 
- model folder : /workdir/models/video_classification/elegant_heisenberg 
- creation time : 11/11/2017 

# Best model - training summary 

## Training set 

|    bird |   blank |   cattle |   chimpanzee |   elephant |   forest buffalo |   gorilla |   hippopotamus |    human |     hyena |   large ungulate |   leopard |       lion |   other (non-primate) |   other (primate) |   pangolin |   porcupine |    reptile |   rodent |   small antelope |   small cat |   wild dog |   duiker |      hog |
|--------:|--------:|---------:|-------------:|-----------:|-----------------:|----------:|---------------:|---------:|----------:|-----------------:|----------:|-----------:|----------------------:|------------------:|-----------:|------------:|-----------:|---------:|-----------------:|------------:|-----------:|---------:|---------:|
| 0.29896 | 0.29626 | 0.106435 |     0.295339 |   0.253236 |       0.00854112 |  0.065666 |      0.0591144 | 0.305003 | 0.0074735 |        0.0723414 | 0.0671436 | 0.00213532 |              0.297919 |          0.307465 |  0.0597877 |    0.146545 | 0.00533825 | 0.301292 |        0.0805357 |   0.0411201 |  0.0202852 | 0.307921 | 0.297373 |

## Validation set 

|     bird |    blank |   cattle |   chimpanzee |   elephant |   forest buffalo |   gorilla |   hippopotamus |   human |     hyena |   large ungulate |   leopard |        lion |   other (non-primate) |   other (primate) |   pangolin |   porcupine |     reptile |   rodent |   small antelope |   small cat |   wild dog |   duiker |      hog |
|---------:|---------:|---------:|-------------:|-----------:|-----------------:|----------:|---------------:|--------:|----------:|-----------------:|----------:|------------:|----------------------:|------------------:|-----------:|------------:|------------:|---------:|-----------------:|------------:|-----------:|---------:|---------:|
| 0.310821 | 0.319279 | 0.096645 |     0.328496 |   0.247319 |       0.00959414 | 0.0446518 |      0.0713553 | 0.30127 | 0.0287824 |        0.0732596 | 0.0646489 | 1.00001e-07 |              0.294846 |          0.327406 |  0.0671588 |    0.148026 | 1.00001e-07 | 0.290823 |        0.0777844 |    0.010056 |  0.0191883 |  0.26873 | 0.294999 |



# Exepriment description 

```python 
{   'callback_config': {   'cycliclr': False,
                           'keep_only_n': 5,
                           'mode': 'min',
                           'monitor': 'val_loss',
                           'patience': 35,
                           'reducelr': True,
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
    'expname': 'exp2',
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
                           'optimizer': 'RMSprop',
                           'optimizer_config': {   'decay': 0.0,
                                                   'epsilon': 1e-08,
                                                   'lr': 0.001,
                                                   'rho': 0.9}}}
```