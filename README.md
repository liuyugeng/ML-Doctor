# ML-Doctor Demo Code
This is the demo code for our USENIX Security 22 paper [ML-Doctor: Holistic Risk Assessment of Inference Attacks Against Machine Learning Models](https://www.usenix.org/conference/usenixsecurity22/presentation/liu-yugeng)

We prefer the users could provide the dataloader by themselves. But we show the demo dataloader in the code. Due to the size of the dataset, we won't upload it to github.

For UTKFace, we have two folders downloaded from [official website](https://susanqq.github.io/UTKFace/) in the UTKFace folder. The first is the "processed" folder which contains three landmark_list files(also can be downloaded from the official website). It is used to get the image name in a fast way because all the features of the images can be achieved from the file names. The second folder is the "raw" folder which contains all the aligned and cropped images. 

For CelebA dataset, we have one folder and three files in the "celeba" folder. For the "img_celeba" folder, it contains all the images downloaded from the [official website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and we align and crop them by ourselves. The others are three files used to get the attributes or file names, named "identity_CelebA.txt", "list_attr_celeba.txt", and "list_eval_partition.txt". The crop center is \[89, 121\] but it is ok if the users wouldn't like to crop it because we have resize function in the transforms so that it will not affect the input shapes.

For FMNIST and STL10, PyTorch has offered the dataset and can be easily employed.

## Preparing
Users should install Python3 and PyTorch at first. If you want to train differential privacy shadow models, you should also install the [opacus](https://github.com/pytorch/opacus).

Or directly run ```pip install -r requirements.txt```

## Testing
```python demo.py --attack_type 0```

<table><tbody>
<!-- TABLE BODY -->
<tr>
<td align="center">Attack Type</td>
<td align="center">0</td>
<td align="center">1</td>
<td align="center">2</td>
<td align="center">3</td>
</tr>
<tr>
<td align="center">Name</td>
<td align="center">MemInf</td>
<td align="center">ModInv</td>
<td align="center">ModSteal</td>
<td align="center">AttrInf</td>
</tr>
</tbody></table>

### For MemInf
We have four modes in this function
<table><tbody>
<!-- TABLE BODY -->
<tr>
<td align="center">Mode</td>
<td align="center">0</td>
<td align="center">1</td>
<td align="center">2</td>
<td align="center">3</td>
</tr>
<tr>
<td align="center">Name</td>
<td align="center">BlackBox Shadow</td>
<td align="center">BlackBox Partial</td>
<td align="center">WhiteBox Partial</td>
<td align="center">WhiteBox Shadow</td>
</tr>
</tbody></table>

#### When buliding attack dataset
When using mode 0 and mode 3, users should choose [```get_attack_dataset_with_shadow```](./doctor/meminf.py#L689) function.
When using mode 1 and mode 2, users should choose [```get_attack_dataset_without_shadow```](./meminf.py#L663) function.

#### When choosing attack model
When using mode 0, ```attack_model``` should be [```ShadowAttackModel```](./utils/define_models.py#L15).
When using mode 1, ```attack_model``` should be [```PartialAttackModel```](./utils/define_models.py#L56).
When using mode 2 and mode 3, ```attack_model``` should be [```WhiteBoxAttackModel```](./utils/define_models.py#L97).
Users can also define attack models by themselves so we didn't fix the models here.

### For ModInv
We have provide two attacks in the pipeline. So no need to modify anything.

### For AttrInf and ModSteal
There are two mode in general, i.e. Partial and Shadow. Users could change the training set in [```main```](./demo.py#L225-L229) function


## Citation
Please cite this paper in your publications if it helps your research:

    @inproceedings {277098,
    title = {{ML-Doctor}: Holistic Risk Assessment of Inference Attacks Against Machine Learning Models},
    booktitle = {31st USENIX Security Symposium (USENIX Security 22)},
    year = {2022},
    address = {Boston, MA},
    url = {https://www.usenix.org/conference/usenixsecurity22/presentation/liu-yugeng},
    publisher = {USENIX Association},
    month = aug,
    }


## License

ML-Doctor is freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, please drop an e-mail at admin@mldoctor.io. We will send the detail agreement to you.
