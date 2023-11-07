# waldo-is-right-there
Author: Richard Li
Training dataset comes from [this lovely repository](https://github.com/vc1492a/Hey-Waldo/tree/master/64/notwaldo)

I ran all of this in a virtual environment, but `pipenv` was being very non-cooperative (see [Challenges](#Challenges)). I ended up just `pip install`ing within the `pipenv`, which I should've been able to avoid via `pipenv install`, but I just didn't have enough time to debug all of my issues.


    What was the goal of your project? Since everyone is doing a different project, you will have to spend some time setting this context.
    How did you solve the problem (i.e., what methods / algorithms did you use and how do they work)? As above, since not everyone will be familiar with the algorithms you have chosen, you will need to spend some time explaining what you did and how everything works.
    Describe a design decision you had to make when working on your project and what you ultimately did (and why)? These design decisions could be particular choices for how you implemented some part of an algorithm or perhaps a decision regarding which of two external packages to use in your project.
    What if any challenges did you face along the way?
    What would you do to improve your project if you had more time?
    Did you learn any interesting lessons for future robotic programming projects? These could relate to working on robotics projects in teams, working on more open-ended (and longer term) problems, or any other relevant topic.


## Goal

This project is a Waldo finder (from Where's Waldo). Given an image where Waldo is present, draw a box around Waldo!

The goal of this project was for me to get a better understanding of what neural networks need in order to run. I've done a lot of work benchmarking the results of these kinds of models, but have never tried implementing them myself. I just wanted to get an idea of implementation process + what part of this is difficult. In particular, I didn't care if I even trained the model myself or anything, since I have no idea how challenging it can be to get a model working, even if I borrow an existing library to do it.

## Solving the problem

A CNN makes sense for this project because we're working primarily with images. Admittedly, I don't really understand what's happening when a CNN is learning, but here's a high-level explanation:

1. We intialize a model with randomized "weights" (or in our case, because we're using transfer learning, less-than-random weights)
2. At every interval, the network evaluates and predicts our target (where Waldo is) before evaluating its error. Over time, it will make predictions that make this error threshhold as low as possible -- in other words, gradient descent!
3. Gradually, over many iterations, we approach a loss that is negligible and determine that our model is now capable of identifying our target pretty much spot on.

From here, I have some eval images in `assets/eval/` that we can test on to find Waldo!

## Design Decision

I was not interested in actually code the muscle of the model itself, so I used an RCNN model out of the box from Tensorflow. I learned that I could use *transfer learning* to sort of borrow the progress an existing model has already made to shorten the amount of time it takes for my model to converge. In the end, I even decided to just steal the weights of an existing waldo-finder model, because I could explain what I needed to train a model but hadn't worked out the dependency hell necessary to make it happen.

## Challenges

I did not expect to have as much using Tensorflow as I did. With it being such a common ML library, I figured that it'd be really easy for me to just plug and play with it, but I couldn't figure out the way that these dependencies needed to work, even after understanding what configuration/checkpoints were used where. 

I'll avoid getting into too much detail, but `pipenv` wasn't working, and `matplotlib`'s visualizer wasn't wroking... basically every dependency that needed to work wasn't working, which is why the scope of this project is so much less impressive than what it really should be. Honestly, this ended up eating up most of my time, and it was both frustrating and scary to pretty much have nothing to show all the way until the end of the project. The fact that this ate up so much of my time also meant that I didn't get to play around with my model as much as I wanted to before the deadline. 

## Improvements

To improve this project, I would like to now write the underlying CNN that is training the data. This would undoubtedly give me more control over the parameters and give me a bunch of opportunities to both optimize and develop a greater understanding of what is actually happening when a machine "learns".

But that's a bit of an aggressive improvement. The smaller improvements I'd like to make all kind of follow the same trend though: writing and implementing various parts of the project myself that I took from the internet. For instance, writing the script to generate the `.tfrecords` binary that the model uses to train or writing the script to generate the annotations file. 

I think I have a good understanding of what needs to happen before I can pull the trigger on training a model myself, so when I have the compute power and time, I'd like to do it myself. 

## Lessons

This was the first time I did a project solely with the intention of just seeing how a black box takes inputs and gives outputs, rather than a project to better understand the black box itself. I've learned that I should start as basic as possible, and copy or hard-code as many things as necessary to get things to work, before trying to generate the various inputs I need. 

For instance, I spent a lot of time trying to write the script to generate the `.tfrecords` file, when I could've just copied a `.tfrecords` file off the internet and figured out the dependency issues earlier.
