# Multitask convolutional neural networks

I honestly think that a lot of times we are underusing our models. Specially convolutional neural networks. Imagenet pre-trained models have amazing architectures and are super powerful - trained to classify images into 1000 categories. We get them and fine tune them to solve our - almost always way simpler - specific classification task, for instance detect sleeve lengths in garments. Then when we have another task, necklines in garments for instance, we train yet a new specific single task model, and so forth and so on. Having all these task specific models slow down pipeline and consume precious/expensive GPU time. What if we train a model to perform all these multiple classification tasks in one go? That's what I'm calling multitask learning. The idea is to have something like this:

<figure style="width: 50%"  class="align-center">
<img src="https://paulo-blog-media.s3-sa-east-1.amazonaws.com/posts/2020-12-31-multitask_learning/app.jpg" alt="">
<figcaption>Dress classified as striped, long sleeves and v-neck, 3 classification instead of just 1!</figcaption>
</figure>


[Here's me talking about it back](https://www.youtube.com/watch?v=p7Oi6HX27Rs&t=12s) in 2017 at PyCon-UK

And [here's the presentation slides](https://static.sched.com/hosted_files/papislatam2019/86/multitask_papis.pdf) I used when presenting at PAPIs.io 2019.

## Single task vs multitask

In my experience, using good pre-trained architectures, plugging multiple classification layers and training all layers yield results similar to using single task networks. The main difference is the speed up in processing. The results below are from a project I did with [EDITED](www.edited.com), a retail tech company, doing 3 classification tasks:

- Multicolor: binary task, if a garment have a clear dominat color or not
- With model: binary task, if the picture contains a human model or not
- Color: multi-class task, classifying main dominant color into a set of 21 possible colors

In terms of accuracy, the results running as single tasks or multi task, as can be seen below:

<figure style="width: 50%"  class="align-center">
<img src="https://paulo-blog-media.s3-sa-east-1.amazonaws.com/posts/2020-12-31-multitask_learning/accuracy.jpg" alt="">
<figcaption>Similar accuracy levels</figcaption>
</figure>

The magic is in terms of processing time. As one could expect, running these in a multitask network is 3 times faster than running them as single task. Basically n times faster, where n is the number of tasks.

<figure style="width: 50%"  class="align-center">
<img src="https://paulo-blog-media.s3-sa-east-1.amazonaws.com/posts/2020-12-31-multitask_learning/processing_time.jpg" alt="">
</figure>

As a nice side effect, training all layers end up generating better embeddings, since the feature vector now takes into account all tasks, generating a more holistic representation of the image. I'll dive a bit deeper in this matter in a future post.

## How to run it

You need to provide a dataframe like the following:

<figure style="width: 75%"  class="align-center">
<img src="https://paulo-blog-media.s3-sa-east-1.amazonaws.com/posts/2020-12-31-multitask_learning/csv.png" alt="">
<figcaption>Dataframe with path to image and labels for each classification task</figcaption>
</figure>

Where we have the path for the images and the label for each one of the tasks.

Then I'm providing a `config.yml` where you set up everything:

- paths: path for the aforementioned dataframe and where to save the model
- model: here you define the input size, batch size, learning rate and, very important, the target encoding rules for each task.
  - It is a dictionary-like structure where each key is a classification task and each value is a list with the possible labels of the respective task.
  - For instance: `sleeves: ["sleeveless", "long_sleeve", "short_sleeve"]` means that you have a task to classify sleeves and the labels are sleeveless, long_sleeve and short_sleeve.

Then you can run `python train_model.py` and here's what will happen:

- Based on the encoding policy in your `config.yml`, it will one-hot-encode the labels
- An `ImageDataGenerator` will be created to perform data augmentation and to split data between train and test.
- This generator will go through the images on the dataframe and yield `X` and `y`, where `X` is a batch and `y` are a the respective one-hot-encoded labels for all the tasks described in the `config.yml` file.
- Based on the number of tasks and labels for each task, a model architecture will be build, using a pre-trained `Xception` model as core and plugging `n` classifiers, according to what is defined in `config.yml`.
- Model will be trained using `EarlyStopping`, `ReduceLROnPlateau`, `CSVLogger`
  - `ReduceLROnPlateau`: If `validation loss` stops to reduce in a given number of epochs, reduces the learning rate
  - `EarlyStopping`: If `validation loss` stops to reduce in a given number of epochs (larger then the number used on the `ReduceLROnPlateau`) epochs, stops training and saves best model
  - `CSVLogger`: Saves results at the end of each epoch to a CSV file

That's basically it!

## Simple app to check results

[Streamlit] is the best tool to quickly build an app to test our model. Running `streamlit run front_end.py` will start a webserver on your `localhost:8501` where you can paste an image url and check the model prediction, like on the screenshot below:

<figure style="width: 75%"  class="align-center">
<img src="https://paulo-blog-media.s3-sa-east-1.amazonaws.com/posts/2020-12-31-multitask_learning/app.jpg" alt="">
<figcaption>Dress classified as striped, long sleeves and v-neck!</figcaption>
</figure>

## If you want to run an example

EDITED's dataset is proprietary, but you can use the [DeepFashion dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html), which is also fashion related, labeled and have enough data to run a test. I prepared a `process_raw_files.py` and a `dataset_config.yml` to deal with this specific dataset. Download the clothes images and the annotation files and put them all in the `data` folder. You should have something like this:

```
multitask_learning/
├── data/
│   ├── Anno_coarse/
│   ├── Anno_fine/
│   ├── Eval/
│   └── img/
```

Then run `python process_raw_files.py` and it should generate a folder called `img_crop` and a `processed_subset.csv`. Then you should  be able to run `python train_model.py` and, after you have a trained model, `streamlit run frontend.py`.

## But I don't have a GPU, it will take forever!

Ah yes, that's that... So what you can do is clone this repository and copy your data to a session of [google colab](https://colab.research.google.com/). Use a GPU-enabled runtime and you should be good to go! For free!! You can run it from the notebook with `!python train_model.py`. When the model finishes training, remember to download the trained model.

## Closing thoughts

When I first tried this, back in 2016, it was really tricky to do, since at that time I had to build my own iterator to yield multiple outputs. But I had to do it since the GPU cost and processing time was holding back the deployment of new computer vision models. Now the `flow_from_dataframe` method gives you out ouf the box multiple outputs, there is no reason not to try it!