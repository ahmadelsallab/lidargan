Environment perception is crucial for safe Automated Driving. While a variety of sensors powers low-level perception,
high-level scene understanding is enabled by computer vision, machine learning, and deep learning, where the main
challenge is in data scarcity. On the one hand, deep learning
models are known for their hunger to data, on the other
hand, it is both complex and expensive to obtain a massively
labeled dataset, especially for non-common sensors, like
LiDAR, Radar and UltraSonic sensors.
Data augmentation techniques could be used to increase
the amount of the labeled data artificially. As simulators
and game engines become more realistic, they become an
efficient source to obtain annotated data for different environments and different conditions. However, there remain two main challenges: 1) For non-common sensors, we need
to develop a model that perceives the simulator environment and produce data in the same way the physical sensor
would do, which we call "Sensor Model" and 2) The sensor
model should consider the realistic noisy conditions in the
natural scenes, unlike the perfect simulation environments,
such as imperfect reflections, and material reflectivity for
LiDAR sensors as an example. Sensors models are hard to
develop in a closed mathematical form. Instead, they better
be learned from real data. However, it is almost impossible
to get a paired dataset, composed of noisy and clean data
for the same scene, because this requires to have the sensor
model in the first place. This dilemma calls for the need
to unsupervised methods, which can learn the model from
unpaired and unlabeled data.
In this work, we formulate the problem of sensor modeling as an image-to-image translation and employ CycleGAN (Zhu et al., 2017a) to learn the mapping. We formulate the two domains to translate between 1) the real LiDAR
point cloud (PCL) and 2) the simulated LiDAR point cloud.
In another setup, we also translate low-resolution LiDAR,
into higher resolution generated LiDAR. For the realistic
LiDAR, we use the KITTI dataset (Geiger et al., 2013),
and for the simulated LiDAR, we use the CARLA simulator (Dosovitskiy et al., 2017). The processing of the input
LiDAR is done in two ways
• Bird-eye View (2D BEV): projection as an OccupancyGrid Map (OGM)
• Polar-Grid Map (2D PGM): where the LiDAR PCL
is transformed in the polar coordinates, which maps
precisely the physical way the LiDAR sensor scans the
environment. Accordingly, no losses occur where all
points in the PCL are considered. Furthermore, the
reverse mapping from the 2D PGM to the 3D PCL is
possible.
The rest of the paper is organized as follows: first, the
related work is presented, then the approach is formulated
under the umbrella of CycleGANs, including the different
translation domains. Following that, the experimental setup
and the different environments and parameters are discussed.
Finally, we conclude with the discussion and future works,focusing on the different ways to evaluate the generated
data

# CycleGAN

In image-to-image translation the model is trained to map
from images in a source domain to images in a target domain
where conditional GANs (CGANs) are adopted to condition the generator and the discriminator on prior knowledge.
Generally speaking, image-to-image translation can be either supervised or unsupervised. In the supervised setting,
Pix2Pix (Isola et al., 2017), SRGAN (Ledig et al., 2017),
the training data is organized in pairs of input and the corresponding output samples. However, in some cases the
paired training data is not available and only unpaired data
is available. In this case the mapping is learned in an unsupervised way given unpaired data, as in CycleGAN (Zhu
et al., 2017a), UNIT (Liu et al., 2017). Moreover, instead
of generating one image, it is possible to generate multiple images based on a single source image in multi-modal
image translation. Multimodal translation can be paired
Pix2PixHD (Wang et al., 2018b), BicycleGAN (Zhu et al.
In image-to-image translation the model is trained to map
from images in a source domain to images in a target domain
where conditional GANs (CGANs) are adopted to condition the generator and the discriminator on prior knowledge.
Generally speaking, image-to-image translation can be either supervised or unsupervised. In the supervised setting,
Pix2Pix (Isola et al., 2017), SRGAN (Ledig et al., 2017),
the training data is organized in pairs of input and the corresponding output samples. However, in some cases the
paired training data is not available and only unpaired data
is available. In this case the mapping is learned in an unsupervised way given unpaired data, as in CycleGAN (Zhu
et al., 2017a), UNIT (Liu et al., 2017). Moreover, instead
of generating one image, it is possible to generate multiple images based on a single source image in multi-modal
image translation. Multimodal translation can be paired
Pix2PixHD (Wang et al., 2018b), BicycleGAN (Zhu et al.

# LiDAR CycleGAN
To enable LiDAR sensor modeling and data
augmentation, unsupervised image-to-image translation is
adapted to map between 2D LiDAR domains where the
complex 3D point cloud processing is avoided. Two LiDAR
domains are used; simulated LiDAR from CARLA urban
car simulator and KIITI dataset
The main idea is to translate LiDAR from a source domain
to another target domain. In the following experiments, two
2D LiDAR representations are used. The first is the 2D
bird-view, where the 3D point cloud is projected where the
height view is lost. The second is the PGM method that
represents the LiDAR 3D point cloud into a 2D grid that
encodes both channels and the ray step angle of the LiDAR
sensor. The 3D point cloud can be reconstructed from the
PGM 2D representation given enough horizontal angular
resolution.

# LiDAR translations
## Sim2Real
 The first experiment is to perform Imageto-Image LiDAR BEV translation from the CARLA simulator frames to the realistic KITTI frames and vice versa
 
 ### PGM
 we test another projection
method of LiDAR in the polar coordinate system, generating a Polar Grid Map (PGM). The way the LiDAR scans the
environment is by sending multiple laser rays in the vertical
direction, which define the number of LiDAR channels or
layers. Each beam scans a plan, in the horizontal radial
direction with a specific angular resolution. A LiDAR sensor configuration can be defined in terms of the number of
channels, the angular resolution of the rays, in addition to
its Field of View (FoV). PGM takes the 3D LiDAR point
cloud and fit it into a 2D grid of values. The rows of the
2D grid represent the number of channels of the sensor (For
example, 64 or 32 in case of Velodyne). The columns of
the 2D grid represent the LiDAR ray step in the radial direction, where the number of steps equals the FoV divided
by the angular resolution. The value of each cell is ranging
from 0.0 to 1.0 describing the distance of the point from
the sensor. PGM translations are shown in figure 7. Such
an encoding has many advantages. First, all the 3D LiDAR
points are included in the map. Second, there is no memory
overhead, like Voxelization for instance. Third, the resulting
representation is dense, not sparse

## Real2Real
: In another experiment, we test the translation between different sensors models configurations. In
particular, we test the ability of CycleGAN to translate from
a domain with few LiDAR channels to another domain with
denser channels
## Sensor2Sensor
CAM2LIDAR

# Evaluation of the generated LiDAR quality

There are number of approaches for evaluating GANs. For
example, Inception Score (IS) (Salimans et al., 2016) provides a method to evaluate both the quality and diversity of
the generated samples. IS is found to be well correlated with
scores from human annotators. In Fréchet Inception Distance (FID) (Heusel et al., 2017), the samples are embeddedinto a feature space and modeled as a continuous multivariate Gaussian distribution, where the Fréchet distance is
evaluated between the fake and the real data. Thus far, the
only way to evaluate the quality of the generated LiDAR
is either subjective, through human judgment, or intrinsic,
through the quality of the reconstruction in sim2real setup
(CARLA2KITTI), or real2sim setup (KITTI2CARLA). In
this work, we present another method, which is "annotation
transfer" in figure 8, by viewing the object bounding boxes
from the annotations in one domain, and project it on the
generated image in the new domain. Again, this method
can serve as visual assessment, and also subject to human
judgment.

# Extrinsic evaluation of the generated LiDAR quality
As a step towards a quantitative extrinsic assessment, one
can think of using an external "judge," that evaluates the
quality of the generated LiDAR in terms of the performance
on a specific computer vision task, object detection for instance. The evaluation can be conducted using a standard
benchmark data set YObj , which is manually annotated like
KITTI. For that purpose, an object detection network from
LiDAR is trained, For example (Ali et al., 2018) and (El Sallab et al., 2018). Starting from simulated data X, we can
get the corresponding generated Y data using G. We have
the corresponding simulated data annotations YObj . Object
detection network can be used to obtain YˆObj , which can be
compared in terms of mIoU to the exact YObj . Moreover,
Object detection network can be initially trained on the real
benchmark data, KITTI, and then augment the data with the
simulator generated data, and the mIoU can be monitored
continuously to measure the improvement achieved by the
data augmentation. The entire evaluation pipeline is shown
in figure 9.
In essence, we want to ensure that, the essential scene objects are mapped correctly when transferring the style of
the realistic LiDAR. This can be thought as similar to the
content loss in the style transfer networks. To ensure this
consistency, an extrinsic evaluation model is used. We call
this model a reference model. The function of this model is
to map the input LiDAR (X or Y ) to a space that reflects the
objects in the scene, like semantic segmentation or object
detection networks. For instance, this network could be
YOLO network trained to detect objects from LiDAR views.
More information is provided in the Appendix section 7.

# Task specific loss
In this part, we want to augment the cycle consistency losses
LRX and LRY with another loss that evaluates the object information existence in the mapped LiDAR from simulation
to realistic.

# Conclusion
In this work, we tackled the problem of LiDAR sensor modeling in simulated environments, to augment real data with
synthetic data. The problem is formulated as an imageto-image translation, where CycleGAN is used to map
sim2real and real2real. KITTI is used as the source of
real LiDAR PCL, and CARLA as the simulation environment. The experimental results showed that the network
is able to translate both BEV and PGM representations of
LiDAR. Moreover, the intrinsic metrics, like reconstruction
loss, and visual assessments show a high potential of the
proposed solution. We discuss in the future works a method
to evaluate the quality of the generated LiDAR, and also
we presented an algorithm to extend the Vanilla CycleGAN
with a task-specific loss, to ensure the correct mapping of
the critical content while doing the translation.

# Data augmentation
- YOLO2D = train(KITTI), eval(KITTI)
- x, Y = CARLA()
- X, Y = G(X, Y)
- YOLO2D = finetune(X, Y), eval(KITTI)
