# ML-Doctor Demo Code
This is the demo code for our USENIX Security 22 paper [ML-Doctor: Holistic Risk Assessment of Inference Attacks Against Machine Learning Models](https://www.usenix.org/conference/usenixsecurity22/presentation/liu-yugeng)

We prefer the users could provide the dataloader by themselves. But we show the demo dataloader in the code. Due to the size of the dataset, we won't upload it to github.

For UTKFace, we have two folders downloaded from [official website](https://susanqq.github.io/UTKFace/) in the UTKFace folder. The first is the "processed" folder which contains three landmark_list files(also can be downloaded from the official website). It is used to get the image name in a fast way because all the features of the images can be achieved from the file names. The second folder is the "raw" folder which contains all the aligned and cropped images. 

For CelebA dataset, we have one folder and three files in the "celeba" folder. For the "img_celeba" folder, it contains all the images downloaded from the [official website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and we align and crop them by ourselves. The others are three files used to get the attributes or file names, named "identity_CelebA.txt", "list_attr_celeba.txt", and "list_eval_partition.txt". The crop center is \[89, 121\] but it is ok if the users wouldn't like to crop it because we have resize function in the transforms so that it will not affect the input shapes.

For FMNIST and STL10, PyTorch has offered the dataset and can be easily employed.

# Preparing
Users should install Python3 and PyTorch at first. If you want to train differential privacy shadow models, you should also install the [opacus](https://github.com/pytorch/opacus).

# Testing
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
<td align="center">Membership Inference</td>
<td align="center">Model Inversion</td>
<td align="center">Model Stealing</td>
<td align="center">Attribute Inference</td>
</tr>
</tbody></table>