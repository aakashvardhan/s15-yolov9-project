
# Training Sketch GUI Images using YOLOv9

## Introduction

This project is about training a YOLOv9 model to detect objects in sketch images. The dataset used for training is a custom dataset from RoboFlow. The dataset contains images of GUI sketches and the corresponding labels. The labels are in the YOLO format. The dataset is divided into training and validation sets. The training set contains 934 images and the validation set contains 152 images. The dataset is available in the `yolov9/data/sketch_images` directory.

## YOLOv9

YOLOv9

YOLOv9 is the latest iteration in the YOLO (You Only Look Once) series, designed for real-time object detection. Released on February 21, 2024, YOLOv9 introduces several groundbreaking innovations to enhance efficiency, accuracy, and adaptability in computer vision tasks.

### Key Features

#### Programmable Gradient Information (PGI)
PGI addresses the information bottleneck problem by preserving essential data across network layers. This ensures accurate gradient updates and enhances the model's learning efficiency.

#### Generalized Efficient Layer Aggregation Network (GELAN)
GELAN optimizes parameter utilization and computational efficiency, making YOLOv9 suitable for both lightweight and large-scale models. This architecture improves the model's ability to retain information and perform accurate predictions.

#### Reversible Functions
Reversible functions help mitigate information loss by allowing operations to reverse their inputs back to their original form. This approach ensures that crucial data is not lost during transformations within the network.

### Supported Tasks and Modes

YOLOv9 supports various tasks, including:
- **Object Detection**
- **Instance Segmentation**

The model is compatible with multiple operational modes such as:
- **Inference**
- **Validation**
- **Training**
- **Export**

### Model Variants

YOLOv9 comes in four main variants, each optimized for different computational needs and accuracy requirements:
- **YOLOv9-S** : Small model with fewer parameters and faster inference speed.
- **YOLOv9-M** : Medium-sized model with a balance between speed and accuracy.
- **YOLOv9-C** : Large model with higher accuracy and more parameters.
- **YOLOv9-E** : Extra-large model with the highest accuracy and computational cost.

### Performance

YOLOv9 outperforms previous YOLO models and other state-of-the-art object detectors on the MS COCO dataset. It achieves higher mean Average Precision (mAP) while using fewer parameters and computational resources.

## Training Result




```

train_dual: weights=/home/ubuntu/s15-yolov9-project/weights/yolov9-e.pt, cfg=/home/ubuntu/s15-yolov9-project/weights/yolov9-e.yaml, data=/home/ubuntu/s15-yolov9-project/yolov9/data/sketch_images/data.yaml, hyp=/home/ubuntu/s15-yolov9-project/yolov9/data/hyps/hyp.scratch-high.yaml, epochs=50, batch_size=8, imgsz=640, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, noplots=False, evolve=None, bucket=, cache=None, image_weights=False, device=0, multi_scale=False, single_cls=False, optimizer=SGD, sync_bn=False, workers=8, project=yolov9/runs/train, name=exp, exist_ok=False, quad=False, cos_lr=False, flat_cos_lr=False, fixed_lr=False, label_smoothing=0.0, patience=100, freeze=[0], save_period=-1, seed=0, local_rank=-1, min_items=0, close_mosaic=0, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest
YOLO 🚀 v0.1-104-g5b1ea9a Python-3.11.9 torch-2.3.0 CUDA:0 (Tesla T4, 14931MiB)

34m hyperparameters: lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, cls_pw=1.0, obj=0.7, obj_pw=1.0, dfl=1.5, iou_t=0.2, anchor_t=5.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.9, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.15, copy_paste=0.3

                 from  n    params  module                                  arguments                     
  0                -1  1         0  models.common.Silence                   []                            
  1                -1  1      1856  models.common.Conv                      [3, 64, 3, 2]                 
  2                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  3                -1  1    252160  models.common.RepNCSPELAN4              [128, 256, 128, 64, 2]        
  4                -1  1    164352  models.common.ADown                     [256, 256]                    
  5                -1  1   1004032  models.common.RepNCSPELAN4              [256, 512, 256, 128, 2]       
  6                -1  1    656384  models.common.ADown                     [512, 512]                    
  7                -1  1   4006912  models.common.RepNCSPELAN4              [512, 1024, 512, 256, 2]      
  8                -1  1   2623488  models.common.ADown                     [1024, 1024]                  
  9                -1  1   4269056  models.common.RepNCSPELAN4              [1024, 1024, 512, 256, 2]     
 10                 1  1      4160  models.common.CBLinear                  [64, [64]]                    
 11                 3  1     49344  models.common.CBLinear                  [256, [64, 128]]              
 12                 5  1    229824  models.common.CBLinear                  [512, [64, 128, 256]]         
 13                 7  1    984000  models.common.CBLinear                  [1024, [64, 128, 256, 512]]   
 14                 9  1   2033600  models.common.CBLinear                  [1024, [64, 128, 256, 512, 1024]]
 15                 0  1      1856  models.common.Conv                      [3, 64, 3, 2]                 
 16[10, 11, 12, 13, 14, -1]  1         0  models.common.CBFuse                    [[0, 0, 0, 0, 0]]             
 17                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
 18[11, 12, 13, 14, -1]  1         0  models.common.CBFuse                    [[1, 1, 1, 1]]                
 19                -1  1    252160  models.common.RepNCSPELAN4              [128, 256, 128, 64, 2]        
 20                -1  1    164352  models.common.ADown                     [256, 256]                    
 21  [12, 13, 14, -1]  1         0  models.common.CBFuse                    [[2, 2, 2]]                   
 22                -1  1   1004032  models.common.RepNCSPELAN4              [256, 512, 256, 128, 2]       
 23                -1  1    656384  models.common.ADown                     [512, 512]                    
 24      [13, 14, -1]  1         0  models.common.CBFuse                    [[3, 3]]                      
 25                -1  1   4006912  models.common.RepNCSPELAN4              [512, 1024, 512, 256, 2]      
 26                -1  1   2623488  models.common.ADown                     [1024, 1024]                  
 27          [14, -1]  1         0  models.common.CBFuse                    [[4]]                         
 28                -1  1   4269056  models.common.RepNCSPELAN4              [1024, 1024, 512, 256, 2]     
 29                 9  1    787968  models.common.SPPELAN                   [1024, 512, 256]              
 30                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 31           [-1, 7]  1         0  models.common.Concat                    [1]                           
 32                -1  1   4005888  models.common.RepNCSPELAN4              [1536, 512, 512, 256, 2]      
 33                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 34           [-1, 5]  1         0  models.common.Concat                    [1]                           
 35                -1  1   1069056  models.common.RepNCSPELAN4              [1024, 256, 256, 128, 2]      
 36                28  1    787968  models.common.SPPELAN                   [1024, 512, 256]              
 37                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 38          [-1, 25]  1         0  models.common.Concat                    [1]                           
 39                -1  1   4005888  models.common.RepNCSPELAN4              [1536, 512, 512, 256, 2]      
 40                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 41          [-1, 22]  1         0  models.common.Concat                    [1]                           
 42                -1  1   1069056  models.common.RepNCSPELAN4              [1024, 256, 256, 128, 2]      
 43                -1  1    164352  models.common.ADown                     [256, 256]                    
 44          [-1, 39]  1         0  models.common.Concat                    [1]                           
 45                -1  1   3612672  models.common.RepNCSPELAN4              [768, 512, 512, 256, 2]       
 46                -1  1    656384  models.common.ADown                     [512, 512]                    
 47          [-1, 36]  1         0  models.common.Concat                    [1]                           
 48                -1  1  12860416  models.common.RepNCSPELAN4              [1024, 512, 1024, 512, 2]     
 49[35, 32, 29, 42, 45, 48]  1  10993616  models.yolo.DualDDetect                 [8, [256, 512, 512, 256, 512, 512]]
yolov9-e summary: 1475 layers, 69418640 parameters, 69418608 gradients, 244.9 GFLOPs

Transferred 2160/2172 items from /home/ubuntu/s15-yolov9-project/weights/yolov9-e.pt
[34m[1mAMP: [0mchecks passed ✅
[34m[1moptimizer:[0m SGD(lr=0.01) with parameter groups 356 weight(decay=0.0), 375 weight(decay=0.0005), 373 bias
[34m[1malbumentations: [0mBlur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01), CLAHE(p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))
[34m[1mtrain: [0mScanning /home/ubuntu/s15-yolov9-project/yolov9/data/sketch_images/train/[0m
[34m[1mval: [0mScanning /home/ubuntu/s15-yolov9-project/yolov9/data/sketch_images/valid/la[0m
Plotting labels to yolov9/runs/train/exp2/labels.jpg... 
Image sizes 640 train, 640 val
Using 8 dataloader workers
Logging results to [1myolov9/runs/train/exp2[0m
Starting training for 50 epochs...

             0/49      13.7G      1.568      2.192      1.484        375        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516       0.71      0.589      0.617       0.46

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/49      13.7G      1.245     0.8649      1.267        336        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.796      0.782      0.847      0.588

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/49      13.7G      1.303       0.85       1.27        423        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.856      0.871      0.884      0.629

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/49      13.7G      1.319     0.7838       1.26        627        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.793      0.806      0.834      0.588

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/49      13.7G       1.25     0.7012      1.255        395        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.884      0.838      0.877      0.618

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       5/49      13.8G      1.222      0.666      1.243        241        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.843      0.872      0.918      0.652

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/49      13.8G      1.204     0.6591      1.241        398        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516       0.85      0.895        0.9      0.609

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       7/49      13.8G      1.191     0.6408      1.241        387        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.884      0.851      0.881      0.628

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       8/49      15.3G       1.18     0.6207      1.226        421        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516       0.91      0.897      0.923      0.684

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       9/49      15.3G      1.138     0.6113       1.22        228        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.844       0.89       0.89      0.656

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      10/49      15.3G      1.132     0.6001      1.212        299        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516       0.93      0.903      0.919      0.666

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      11/49      15.3G      1.113     0.5867      1.216        199        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.906      0.888      0.917      0.656

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      12/49      15.3G      1.106     0.5806      1.211        312        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.876      0.914      0.911      0.663

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      13/49      15.3G      1.098     0.5703      1.201        225        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.863      0.901      0.903       0.66

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      14/49      15.3G      1.089     0.5695      1.202        220        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.932      0.899      0.915      0.668

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      15/49      15.3G      1.076     0.5527        1.2        426        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516       0.92      0.912      0.929      0.672

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      16/49      15.3G      1.074     0.5429      1.196        315        640:  

            16/49      15.3G      1.072     0.5488      1.197        443        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.937      0.887      0.925      0.664

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      17/49      15.3G      1.069     0.5532      1.193        302        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.871      0.894      0.893      0.654

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      18/49      15.3G      1.055     0.5393       1.19        352        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516       0.91      0.893      0.915      0.675

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      19/49      15.3G      1.055      0.535      1.193        253        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.924      0.886      0.915      0.673

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      20/49      15.3G      1.038      0.521      1.184        389        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.929      0.895      0.924      0.682

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      21/49      15.3G      1.041      0.528      1.184        259        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.924      0.903       0.91      0.661

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      22/49      15.3G      1.024     0.5197      1.184        384        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.909      0.912      0.922      0.688

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      23/49      15.3G      1.023     0.5195      1.182        395        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.952      0.882      0.927      0.679

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      24/49      15.3G      1.016     0.5067      1.184        319        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.905      0.905      0.932      0.673

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      25/49      15.3G       1.01     0.5058       1.17        375        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.906      0.934      0.926      0.695

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      26/49      15.3G      1.007     0.5046      1.173        364        640:  


            26/49        14G      1.008     0.5013      1.178        343        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.889      0.908      0.906      0.675

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      27/49        14G      1.007     0.5031      1.171        469        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.926      0.871      0.908      0.667

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      28/49        14G     0.9981     0.4931      1.169        232        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.928       0.88      0.926      0.697

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      29/49        14G      1.005     0.5047      1.168        277        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.942      0.917       0.94      0.697

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      30/49        14G     0.9863      0.491      1.169        206        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.954      0.923      0.931      0.687

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      31/49        14G     0.9833     0.4861      1.174        294        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.906      0.884      0.931       0.69

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      32/49        14G     0.9824     0.4853      1.166        281        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.939      0.888      0.931      0.695

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      33/49        14G     0.9794     0.4817      1.168        370        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.916      0.891      0.916      0.688

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      34/49        14G     0.9678     0.4733      1.165        329        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.941      0.905       0.92      0.679

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      35/49        14G     0.9723     0.4734      1.166        472        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.906      0.895      0.935      0.697

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      36/49        14G     0.9707     0.4698      1.158        294        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.941      0.928       0.93      0.692

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      37/49        14G     0.9638     0.4753      1.157        357        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.917      0.919      0.922      0.672

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      38/49        14G     0.9516     0.4579      1.154        457        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.933       0.91      0.932      0.704

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      39/49        14G     0.9503     0.4604      1.149        424        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.905      0.935      0.932      0.697

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      40/49        14G      0.946     0.4526      1.155        157        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.935      0.912      0.932      0.699

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      41/49        14G     0.9334     0.4478      1.151        344        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.956      0.913      0.935      0.705

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      42/49        14G     0.9392     0.4494      1.158        345        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.945       0.92      0.939      0.706

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      43/49        14G     0.9415     0.4437      1.153        623        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.931      0.905      0.927      0.703

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      44/49        14G      0.926     0.4378      1.147        316        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.943      0.928      0.931      0.697

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      45/49        14G     0.9288      0.441      1.145        463        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.908      0.948      0.929      0.696

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      46/49        14G      0.927     0.4401      1.147        337        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.913      0.898      0.933        0.7

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      47/49        14G      0.923     0.4366      1.148        549        640: 1
                 Class     Images  Instances          P          R      mAP50   
                   all        152       4516      0.945      0.925      0.942      0.707

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      48/49        14G     0.9282     0.4242      1.151        369        640:  

      ```


## Inference

To run inference on a single image, use the following command:

```bash
python yolov9/detect_dual.py --source yolov9/data/sketch_images/valid/images/1EACABD4-EF65-46D0-82D3-F01EDCCA4992_png.rf.b6ff1482b613270be937c736f62be716.jpg --img 640 --device cpu --weights yolov9/runs/train/exp2/weights/best.pt --name gui_output
```

Image Result:

![image](https://github.com/aakashvardhan/s15-yolov9-project/blob/main/asset/output.png)

The output image shows the bounding boxes around the GUI Elements. The bounding boxes are color-coded based on the class of the object. The class labels and confidence scores are also displayed on the image.

## Conclusion

The YOLOv9 model was trained on the GUI dataset, and the model was able to detect the GUI elements with high accuracy. The model was trained for 50 epochs, and the best model was selected based on the validation loss. The model was able to achieve an mAP of 0.942 on the validation set. The model was then used to perform inference on a sample image, and the model was able to detect the GUI elements in the image with high accuracy.